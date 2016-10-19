import os
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

import numpy as np
import planning.transformations as tf

path = os.path.dirname(os.path.abspath(__file__))


class Kinematics():
    def __init__(self):
        # self.robot = URDF.from_parameter_server()
        self.robot = URDF.from_xml_file(
            os.path.join(path, '../../../cyplam_workcell/urdf/workcell.urdf'))

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
