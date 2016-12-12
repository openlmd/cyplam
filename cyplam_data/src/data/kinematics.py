from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

import numpy as np
import planning.transformations as tf


class Kinematics():
    def __init__(self):
        self.robot = URDF.from_parameter_server()
        #self.robot = URDF.from_xml_file('../../../../mashes/mashes_workcell/urdf/workcell.urdf')

    def get_tree(self):
        tree = kdl_tree_from_urdf_model(self.robot)
        print tree.getNrOfSegments()

    def get_chain(self, tree):
        chain = tree.getChain('base_link', 'tool0')
        print chain.getNrOfJoints()

    def set_kinematics(self, base='base_link', target='tcp0'):
        self.kdl_kin = KDLKinematics(self.robot, base, target)

    def get_random_joints(self):
        q = self.kdl_kin.random_joint_angles()
        return q

    def get_pose(self, joints):
        # forward kinematics (returns homogeneous 4x4 numpy.mat)
        pose = self.kdl_kin.forward(joints)
        quat = tf.quaternion_from_matrix(pose)
        trans = np.squeeze(np.asarray(pose[:3, 3]))
        return trans, quat


if __name__ == "__main__":
    kdl = Kinematics()
    kdl.set_kinematics('base_link', 'tcp0')
    joints = kdl.get_random_joints()
    print 'pose:', kdl.get_pose(joints)


# import random

# robot = URDF.from_parameter_server()

# base_link = robot.get_root()
# end_link = robot.link_map.keys()[random.randint(0, len(robot.link_map)-1)]
# print "Root link: %s; Random end link: %s" % (base_link, end_link)
# kdl_kin = KDLKinematics(robot, base_link, end_link)
# q = kdl_kin.random_joint_angles()
# print "Random angles:", q
# pose = kdl_kin.forward(q)
# print "FK:", pose

# q_new = kdl_kin.inverse(pose)
# print "IK (not necessarily the same):", q_new
# if q_new is not None:
#     pose_new = kdl_kin.forward(q_new)
#     print "FK on IK:", pose_new
#     print "Error:", np.linalg.norm(pose_new * pose**-1 - np.mat(np.eye(4)))
# else:
#     print "IK failure"

# J = kdl_kin.jacobian(q)
# print "Jacobian:", J
# M = kdl_kin.inertia(q)
# print "Inertia matrix:", M
# if False:
#     M_cart = kdl_kin.cart_inertia(q)
#     print "Cartesian inertia matrix:", M_cart
