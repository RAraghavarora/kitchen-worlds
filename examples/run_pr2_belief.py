#!/usr/bin/env python
from __future__ import print_function
from os.path import join
# import sys
# sys.path.append("/home2/raghav.arora/llm_tamp/kitchen-worlds/pybullet_planning/pybullet_tools")
# sys.path.append("/home2/raghav.arora/llm_tamp/")
# sys.path.append("/home2/raghav.arora/llm_tamp/kitchen-worlds/pybullet_planning/")
# sys.path.append("/home2/raghav.arora/llm_tamp/kitchen-worlds/pddlstream")
# sys.path.append("/home2/raghav.arora/llm_tamp/kitchen-worlds/")
# import pybullet_tools
import sys

# Uncomment this when running for the first time to get a trace of function calls
# import tracit
# sys.setprofile(tracit.tracefunc)

try:
    import pybullet as p
except ImportError:
    raise ImportError('This example requires PyBullet (https://pypi.org/project/pybullet/)')

import config
from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.algorithms.search import ABSTRIPSLayer
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test, accelerate_list_gen_fn
from pddlstream.utils import read, get_file_path, Profiler
from pddlstream.language.constants import PDDLProblem, And, Equal, print_solution
from pddlstream.language.stream import StreamInfo

# from pybullet_planning.pybullet_tools.utils import set_pose, get_pose, connect, clone_world, \
#     disconnect, set_client, add_data_path, WorldSaver, wait_for_user, get_joint_positions, get_configuration, \
#     set_configuration, ClientSaver, HideOutput, is_center_stable, add_body_name, draw_base_limits, VideoSaver

# Remove pybullet_planning from sys path for imports (because examples exists inside pybullet_planning also)
# sys.path.remove(config.join(config.PROJECT_DIR, 'pybullet_planning'))
from examples.pybullet.pr2_belief.primitives import Scan, ScanRoom, Detect, Register, \
    plan_head_traj, get_cone_commands, move_look_trajectory, get_vis_base_gen, \
    get_inverse_visibility_fn, get_in_range_test, VIS_RANGE, REG_RANGE
from examples.pybullet.pr2_belief.problems import get_problem1, USE_DRAKE_PR2, create_pr2
from examples.pybullet.utils.pybullet_tools.pr2_utils import ARM_NAMES, get_arm_joints, attach_viewcone, \
    is_drake_pr2, get_group_joints, get_group_conf
from examples.pybullet.utils.pybullet_tools.utils import set_pose, get_pose, connect, clone_world, \
    disconnect, set_client, add_data_path, WorldSaver, wait_for_user, get_joint_positions, get_configuration, \
    set_configuration, ClientSaver, HideOutput, is_center_stable, add_body_name, draw_base_limits, VideoSaver
from examples.pybullet.utils.pybullet_tools.pr2_primitives import Conf, get_ik_ir_gen, get_motion_gen, get_stable_gen, \
    get_grasp_gen, Attach, Detach, apply_commands, Trajectory, get_base_limits
from examples.discrete_belief.run import revisit_mdp_cost, MAX_COST, clip_cost
from examples.pybullet.utils.pybullet_tools.general_streams import get_grasp_list_gen, get_contain_list_gen, sample_joint_position_closed_gen, get_handle_grasp_gen, sample_joint_position_gen
from examples.pybullet.utils.pybullet_tools.mobile_streams import get_ik_fn_old, get_ik_gen_old, get_ik_ungrasp_gen, get_pull_door_handle_motion_gen
from examples.pybullet.utils.pybullet_tools.rag_utils import get_body_joint_position
# Add pybullet_planning after importing
# sys.path.append(config.join(config.PROJECT_DIR, 'pybullet_planning'))

def pddlstream_from_state(state, teleport=False):
    task = state.task
    robot = task.robot
    # TODO: infer open world from task

    exp_dir = '/home2/raghav.arora/KW/pybullet_planning/pddl_domains/'
    domain_path = join(exp_dir, 'pr2_belief_domain.pddl')
    stream_path = join(exp_dir, 'pr2_belief_stream.pddl')
    domain_pddl = read(domain_path)
    stream_pddl = read(stream_path)
    # domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    # stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {
        'base': 'base',
        'left': 'left',
        'right': 'right',
        'head': 'head',
    }

    #base_conf = state.poses[robot]
    base_conf = Conf(robot, get_group_joints(robot, 'base'), get_group_conf(robot, 'base'))
    scan_cost = 1
    init = [
        ('BConf', base_conf),
        ('AtBConf', base_conf),
        Equal(('MoveCost',), 1),
        Equal(('PickCost',), 1),
        Equal(('PlaceCost',), 1),
        Equal(('ScanCost',), scan_cost),
        Equal(('RegisterCost',), 1),
    ]
    holding_arms = set()
    holding_bodies = set()
    for attach in state.attachments.values():
        holding_arms.add(attach.arm)
        holding_bodies.add(attach.body)
        init += [('Grasp', attach.body, attach.grasp),
                 ('AtGrasp', attach.arm, attach.body, attach.grasp)]
    for arm in ARM_NAMES:
       joints = get_arm_joints(robot, arm)
       conf = Conf(robot, joints, get_joint_positions(robot, joints))
       init += [('Arm', arm), ('AConf', arm, conf), ('AtAConf', arm, conf)]
       if arm in task.arms:
            init += [('Controllable', arm)]
            init += [('canpull', arm)]
       if arm not in holding_arms:
           init += [('HandEmpty', arm)]
    
    # init += task.robot_instance.get_init()

    for body in task.get_bodies():
        if body in holding_bodies:
            continue
        # TODO: no notion whether observable actually corresponds to the correct thing
        pose = state.poses[body]
        init += [('Pose', body, pose), ('AtPose', body, pose),
                 ('Observable', pose),
        ]
    init += [('Scannable', body) for body in task.rooms + task.surfaces]
    init += [('Registerable', body) for body in task.movable]
    init += [('Graspable', body) for body in task.movable]
    for body in task.get_bodies():
        supports = task.get_supports(body)
        if supports is None:
            continue
        for surface in supports:
            p_obs = state.b_on[body].prob(surface)
            cost = revisit_mdp_cost(0, scan_cost, p_obs)  # TODO: imperfect observation model
            init += [('Stackable', body, surface),
                     Equal(('LocalizeCost', surface, body), clip_cost(cost))]
            #if is_placement(body, surface):
            if is_center_stable(body, surface):
                if body in holding_bodies:
                    continue
                pose = state.poses[body]
                init += [('Supported', body, pose, surface)]

    for body in task.get_bodies():
        if state.is_localized(body):
            init.append(('Localized', body))
        else:
            init.append(('Uncertain', body))
        if body in state.registered:
            init.append(('Registered', body))

    fridge = 5
    food = 3
    joint = (fridge, 1)
    fridge_region = (fridge, None, 0) #TODO: idk why
    floor = 1
    task.floors = [floor]
    joint_pos = get_body_joint_position(joint)
    # Code to use fridge added by Raghav
    init += [('door', joint)]
    init += [('space', fridge_region)]
    init += [('region', fridge_region)]
    init += [('joint', joint)]
    init += [('door', joint)]
    init += [('staticlink', joint)]
    init += [('isjointto', joint, fridge)]
    init += [('position', joint, joint_pos)]
    init += [('atposition', joint, joint_pos)]
    init += [('isclosedposition', joint, joint_pos)]
    
    for arm in ARM_NAMES:
        joints = get_arm_joints(robot, arm)
        conf = Conf(robot, joints, get_joint_positions(robot, joints))
        init += [('DefaultAConf', arm, conf)]
    # goal2 = And(goal, ('graspedhandle', str(fridge)+'::joint_1'))
    init += [('canmove',)]
    init += [('canpick',)]
    init += [('cangrasphandle',)]
    init += [('containable', food, fridge_region)]
    init += [('canmovebase',)]


    # init2 = task.world.get_facts()

    goal = And(*[('Holding', a, b) for a, b in task.goal_holding] + \
        #    [('On', b, s) for b, s in task.goal_on] + \
        #    [('In', food, fridge_region)] + \
            [('On', food, fridge)] + \
                # [('CanUngrasp',)] + \
           [('Localized', b) for b in task.goal_localized] + \
           [('Registered', b) for b in task.goal_registered])

    PULL_UNTIL = 1.8 #R For some reason
    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(task)),
        'sample-grasp': from_list_fn(get_grasp_gen(task)),
        'inverse-kinematics': from_gen_fn(get_ik_ir_gen(task, teleport=teleport)),
        'plan-base-motion': from_fn(get_motion_gen(task, teleport=teleport)),

        'test-vis-base': from_test(get_in_range_test(task, VIS_RANGE)),
        'test-reg-base': from_test(get_in_range_test(task, REG_RANGE)),

        'sample-vis-base': accelerate_list_gen_fn(from_gen_fn(get_vis_base_gen(task, VIS_RANGE)), max_attempts=25),
        'sample-reg-base': accelerate_list_gen_fn(from_gen_fn(get_vis_base_gen(task, REG_RANGE)), max_attempts=25),
        'inverse-visibility': from_fn(get_inverse_visibility_fn(task)),
        'sample-pose-inside': from_gen_fn(get_contain_list_gen(task)),
        'inverse-reachability': from_gen_fn(get_ik_gen_old(task, verbose=True, visualize=False, ir_only=True)),
        'get-joint-position-open': from_gen_fn(sample_joint_position_gen(task, num_samples=6, p_max=PULL_UNTIL)),
        'sample-handle-grasp': from_gen_fn(get_handle_grasp_gen(task)),
        'inverse-kinematics-grasp-handle': from_gen_fn(get_ik_gen_old(task, ACONF=True)),
        'inverse-kinematics-ungrasp-handle': from_gen_fn(get_ik_ungrasp_gen(task, verbose=False)),
        'plan-base-pull-handle': from_fn(get_pull_door_handle_motion_gen(task))
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


#######################################################


def post_process(state, plan, replan_obs=True, replan_base=False, look_move=False):
    if plan is None:
        return None
    # TODO: refine actions
    robot = state.task.robot
    commands = []
    uncertain_base = False
    expecting_obs = False
    for i, (name, args) in enumerate(plan):
        if replan_obs and expecting_obs:
            break
        saved_world = WorldSaver() # StateSaver()
        if name == 'move_base':
            c = args[-1]
            [t] = c.commands
            # TODO: look at the trajectory (endpoint or path) to ensure fine
            # TODO: I should probably move base and keep looking at the path
            # TODO: I could keep updating the head goal as the base moves along the path
            if look_move:
                # TODO: some sort of new bug here where the trajectory repeats
                new_commands = [move_look_trajectory(state.task, t)]
                #new_commands = [inspect_trajectory(state.task, t), t]
            else:
                new_commands = [t]
            if replan_base:
                uncertain_base = True
        elif name == 'pick':
            if uncertain_base:
                break
            a, b, p, g, _, c = args
            [t] = c.commands
            attach = Attach(robot, a, g, b)
            new_commands = [t, attach, t.reverse()]
        elif name == 'place':
            if uncertain_base:
                break
            a, b, p, g, _, c = args
            [t] = c.commands
            detach = Detach(robot, a, b)
            new_commands = [t, detach, t.reverse()]
        elif name == 'scan':
            o, p, bq, hq, ht = args
            ht0 = plan_head_traj(state.task, hq.values)
            new_commands = [ht0]
            if o in state.task.rooms:
                attach, detach = get_cone_commands(robot)
                new_commands += [attach, ht, ScanRoom(robot, o), detach]
            else:
                new_commands += [ht, Scan(robot, o)]
                #with BodySaver(robot):
                #    for hq2 in ht.path:
                #        st = plan_head_traj(state.task, hq2.values)
                #        new_commands += [st, Scan(robot, o)]
                #        hq2.step()
            # TODO: return to start conf?
        elif name == 'localize':
            r, _, o, _ = args
            new_commands = [Detect(robot, r, o)]
            expecting_obs = True
        elif name == 'register':
            o, p, bq, hq, ht = args
            ht0 = plan_head_traj(state.task, hq.values)
            register = Register(robot, o)
            new_commands = [ht0, register]
            expecting_obs = True
        elif name == 'grasp_handle':
            # add command
            pass
        elif name == 'pull_handle':
            pass
        else:
            raise ValueError(name)
        saved_world.restore()
        for command in new_commands:
            if isinstance(command, Trajectory) and command.path:
                command.path[-1].assign()
        commands += new_commands
    return commands


#######################################################

import numpy as np
from robot_builder.robots import PR2Robot
from pybullet_planning.robot_builder.robot_builders import set_pr2_ready
from pybullet_tools.pr2_utils import set_group_conf
from pybullet_tools.bullet_utils import BASE_LINK, BASE_RESOLUTIONS, draw_base_limits as draw_base_limits_bb, CAMERA_FRAME, CAMERA_MATRIX, EYE_FRAME, BASE_LIMITS
import numpy as np
from pybullet_tools.pr2_primitives import get_base_custom_limits
from world_builder.entities import Camera
def create_pr2_robot(robot_conf, robot_pose, robot=None, base_q=(0, 0, 0), dual_arm=False, resolutions=BASE_RESOLUTIONS, use_torso=True, custom_limits=BASE_LIMITS):
    # copied from pybullet_planning
    # robot = None
    # if robot is None:
    #     robot = create_pr2(use_drake=USE_DRAKE_PR2)
    #     set_pr2_ready(robot, arm=PR2Robot.arms[0], dual_arm=dual_arm)
    #     if len(base_q) == 3:
    #         set_group_conf(robot, 'base', base_q)
    #     elif len(base_q) == 4:
    #         set_group_conf(robot, 'base-torso', base_q)
    #     set_pose(robot, robot_pose)
    #     set_configuration(robot, robot_conf)
    with np.errstate(divide='ignore'):
        weights = np.reciprocal(resolutions)

    if isinstance(custom_limits, dict):
        custom_limits = np.asarray(list(custom_limits.values())).T.tolist()

    # if draw_base_limits:
    #     draw_base_limits_bb(custom_limits)

    robot = PR2Robot(robot, base_link=BASE_LINK,
                     dual_arm=dual_arm, use_torso=use_torso,
                     custom_limits=get_base_custom_limits(robot, custom_limits),
                     resolutions=resolutions, weights=weights)

    # print('initial base conf', get_group_conf(robot, 'base'))
    # set_camera_target_robot(robot, FRONT=True)

    camera = Camera(robot, camera_frame=CAMERA_FRAME, camera_matrix=CAMERA_MATRIX, max_depth=2.5, draw_frame=EYE_FRAME)
    robot.cameras.append(camera)

    ## don't show depth and segmentation data yet
    # if args.camera: robot.cameras[-1].get_image(segment=args.segment)

    return robot

def plan_commands(state, args, profile=True, verbose=True):
    # TODO: could make indices into set of bodies to ensure the same...
    # TODO: populate the bodies here from state and not the real world
    sim_world = connect(use_gui=args.viewer)
    #clone_world(client=sim_world)
    task = state.task
    robot_conf = get_configuration(task.robot)
    robot_pose = get_pose(task.robot)
    with ClientSaver(sim_world):
        with HideOutput():
            robot = create_pr2(use_drake=USE_DRAKE_PR2)
        set_pose(robot, robot_pose)
        set_configuration(robot, robot_conf)
    # robot_instance = create_pr2_robot(robot_conf, robot_pose, robot=robot)
    # world.add_robot(robot_instance)
    # task.robot_instance = robot_instance
    mapping = clone_world(client=sim_world, exclude=[task.robot]) # TODO: TypeError: argument 5 must be str, not bytes
    assert all(i1 == i2 for i1, i2 in mapping.items())
    set_client(sim_world)
    saver = WorldSaver() # StateSaver()

    pddlstream_problem = pddlstream_from_state(state, teleport=args.teleport)
    _, _, _, stream_map, init, goal = pddlstream_problem
    print('Init:', sorted(init, key=lambda f: f[0]))
    if verbose:
        print('Goal:', goal)
        print('Streams:', stream_map.keys())

    stream_info = {
        'test-vis-base': StreamInfo(eager=True, p_success=0),
        'test-reg-base': StreamInfo(eager=True, p_success=0),
    }
    hierarchy = [
        ABSTRIPSLayer(pos_pre=['AtBConf']),
    ]

    with Profiler(field='cumtime', num=10 if profile else None):
        # import pdb; pdb.set_trace()
        solution = solve(pddlstream_problem, algorithm=args.algorithm, unit_costs=args.unit,
                         stream_info=stream_info, hierarchy=hierarchy, debug=False,
                         success_cost=MAX_COST, verbose=verbose)
        plan, cost, evaluations = solution
        if MAX_COST <= cost:
            plan = None
        print_solution(solution)
        print('Finite cost:', cost < MAX_COST)
        commands = post_process(state, plan)
    saver.restore()
    disconnect()
    return commands


#######################################################

def main(time_step=0.01):
    parser = create_parser()
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-viewer', action='store_true', help='Enable the viewer while planning')
    # TODO: argument for selecting prior
    args = parser.parse_args()
    print('Arguments:', args)
    # TODO: nonuniform distribution to bias towards other actions
    # TODO: closed world and open world
    real_world = connect(use_gui=not args.viewer)
    add_data_path()
    task, state = get_problem1(localized='rooms', p_other=0.25) # surfaces | rooms
    for body in task.get_bodies():
        try:
            add_body_name(body)
        except TypeError:
            add_body_name(body.body)
            # import pdb; pdb.set_trace()

    robot = task.robot
    #dump_body(robot)
    assert(USE_DRAKE_PR2 == is_drake_pr2(robot))
    attach_viewcone(robot) # Doesn't work for the normal pr2?
    draw_base_limits(get_base_limits(robot), color=(0, 1, 0))
    #wait_for_user()
    # TODO: partially observable values
    # TODO: base movements preventing pick without look

    # TODO: do everything in local coordinate frame
    # TODO: automatically determine an action/command cannot be applied
    # TODO: convert numpy arrays into what they are close to
    # TODO: compute whether a plan will still achieve a goal and do that
    # TODO: update the initial state directly and then just redraw it to ensure uniqueness
    step = 0
    with VideoSaver('/home2/raghav.arora/pr2_belief/pr2_combined.mp4'):
        while True:
            step += 1
            print('\n' + 50 * '-')
            print(step, state)
            # wait_for_user()
            #print({b: p.value for b, p in state.poses.items()})
            with ClientSaver():
                commands = plan_commands(state, args)
            print()
            if commands is None:
                print('Failure!')
                break
            if not commands:
                print('Success!')
                break
            apply_commands(state, commands, time_step=time_step)

    print(state)
    # wait_for_user()
    disconnect()


if __name__ == '__main__':
    main()
