from mmky import writers
import cv2

if __name__ == '__main__':
    episode = writers.readRobosuite('d:\\code\\mechanicalmonkey\\mmky\\trajectories\\dropping_real\\robosuite_drop_1_1635109424.3289683.hdf5')
    for image in episode['images']:
        cv2.imshow("image", image)
        cv2.waitKey(16)

    # episode = writers.readSimpleNpy('C:\\code\\mechanicalmonkey\\mmky\\trajectories\\cup_pour_simple.npy')
    # for image in episode['images']:
    #     cv2.imshow("image", image)
    #     #print(episode["actions"])
    #     cv2.waitKey(16)
