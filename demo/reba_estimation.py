import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import scipy.io as io
import csv
import glob
# python reba_estimation.py --gpu 0-1 --test_epoch MuCo+COCO.pth.tar --dataset MuCo
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
sys.path.insert(0, osp.join('..', 'tool/Ergonomic'))
from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
from utils.pose_utils import process_bbox, pixel2cam, world2cam, cam2pixel
from utils.vis import vis_keypoints, vis_3d_multiple_skeleton
from utils.reba import reba_calculator, coords_from_other_dataset, get_MPJPE
from utils.reba_from_vicon import reba_coords_from_our_model, joints_comparison, angles_comparison, write_coords_to_csv
#from vicon2coco import load_joints_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--dataset', type=str, dest='dataset')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    assert args.dataset, 'Dataset is required. (H36M, MuCO, or CVRE)'
    return args

def pose_estimation(args, img_file_name, dataset, model, transform, joint_num, joints_name, flip_pairs, skeleton):


    # prepare bbox for each human. [# of images, # of people, 4 coords]
    bbox_list = [[]]
    #csv file titles
    csv_joints_title = ['P1:Head','','','P1:Nose','','','P1:LSHO','','','P1:LELB','','','P1:LWST','','','P1:RSHO','','','P1:RELB','','','P1:RWST','','','P1:LHIP','','','P1:LKNE','','','P1:LAKL','','','P1:RHIP','','','P1:RKNE','','','P1:RAKL','','','P1:LHND','','','P1:RHND','','']
    csv_angles_title = ['neck_angle', 'neck_twisted', 'neck_side_bending', 'trunk_angle', 'trunk_twisted', 'trunk_side_bending', 'leaning', 'legs_down', 'leg_angle', 'upper_arm_angle', 'shoulder_raised', 'upper_arm_abducted', 'lower_arm_angle', 'wrist_angle', 'wrist_bending', 'Score A', 'Score B', 'Score C', 'REBA Score', 'RULA Score']

    # read detected bounding boxes for people from yolov5
    with open(osp.join('../../data/CVRE3D/info', img_file_name,'bbox_list.txt'), 'r') as f:
        lines = f.read().splitlines()
        lines = lines[:-1]
        ith_img = 0
        for line in lines:
            if line == '':
                bbox_list.append([])
                ith_img += 1
            else:
                # convert txt bbox from str to int
                bbox_temp = [int(j) for j in line.split(' ')]
                bbox_list[ith_img].append(bbox_temp)
    f.close()

    root_depth_list = [[0 for i in range(len(bbox_list[j]))] for j in range(len(bbox_list))]
    # prepare root depth list for a picture, the results are from root net
    with open(osp.join('../../data/CVRE3D/info', img_file_name,'root_jd.txt'), 'r') as f:
        lines = f.read().splitlines()
        ith_img = 0
        for line in lines:
            joint_temp = [float(j) for j in line.split(' ')]
            root_depth_list[ith_img][:] = joint_temp[:]# obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)
            ith_img += 1
    f.close()

    assert len(bbox_list) == len(root_depth_list)

    if os.path.exists(osp.join('../output/test_result',img_file_name,'reba_angles_CVRE.csv')):
        os.remove(osp.join('../output/test_result',img_file_name,'reba_angles_CVRE.csv'))
    if os.path.exists(osp.join('../output/test_result',img_file_name,'reba_angles_gt.csv')):
        os.remove(osp.join('../output/test_result',img_file_name,'reba_angles_gt.csv'))
    #if os.path.exists(osp.join('../output/test_result',img_file_name,'reba_angles_comparison.csv')):
    #    os.remove(osp.join('../output/test_result',img_file_name,'reba_angles_comparison.csv'))

    if os.path.exists(osp.join('../output/test_result',img_file_name,'coords_CVRE.csv')):
        os.remove(osp.join('../output/test_result',img_file_name,'coords_CVRE.csv'))
    if os.path.exists(osp.join('../output/test_result',img_file_name,'coords_gt.csv')):
        os.remove(osp.join('../output/test_result',img_file_name,'coords_gt.csv'))

    # read vicon joints coords and calculate reba
    with open(osp.join('../../data/CVRE3D/info', img_file_name,'joint_pos.csv'), 'r') as f:
        reader = csv.reader(f)
        frames = list(reader)
    f.close()

    images = []
    ith_img = 0
    kpt_estimation_all = np.zeros((len(bbox_list), len(bbox_list[0]),16, 3))
    kpt_gt_all = np.zeros((len(bbox_list), len(bbox_list[0]),16, 3))
    reba_angles_pred = np.zeros((len(bbox_list), len(bbox_list[0]), 20))
    reba_angles_gt = np.zeros((len(bbox_list), len(bbox_list[0]), 20))
    reba_score_pred = np.zeros((len(bbox_list), len(bbox_list[0])))
    reba_score_gt = np.zeros((len(bbox_list), len(bbox_list[0])))

    reba_angles_file = open(osp.join('../output/test_result',img_file_name,'reba_angles_CVRE.csv'), 'a')
    writer = csv.writer(reba_angles_file)
    writer.writerow(csv_angles_title)
    reba_vicon_file = open(osp.join('../output/test_result',img_file_name,'reba_angles_gt.csv'), 'a')
    writer1 = csv.writer(reba_vicon_file)
    writer1.writerow(csv_angles_title)
    files = sorted(glob.glob(os.path.join('../../data/CVRE3D/images', img_file_name, '*.*')))  # dir
    pic = 0

    trained_set = args.test_epoch[:-8]
    subject = img_file_name[5:7]
    if subject == '01':
        # normalized camera intrinsics
        #princpt = [907.112, 477.877]
        focal = [1253.806, 1253.806]# x-axis, y-axis
        princpt = [586.667, 293.333]# princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
    elif subject == '02':
        focal = [1254.016, 1254.016]
        princpt = [613.333, 300.0]
    elif subject == '03':
        focal = [1254.246, 1254.246]
        princpt = [600.0, 296.667]
    elif subject == '04':
        focal = [1253.147, 1253.147]
        princpt = [600.0, 303.333]
    elif subject == '05':
        focal = [1253.031, 1253.031]
        princpt = [600.0, 296.667]
    elif subject == '06':
        focal = [1253.489, 1253.489]
        princpt = [590.0, 300.0]
    else:
        focal = [1351.351, 1353.462]#7.76x5.81mm cmos, 1.924um x 1.921um pixel size
        princpt = [2016, 1512]
        print('Unidentified Subject. Check Line 145.')
    mpjpe_total = 0
    for filename in files:
        pic += 1
        original_img = cv2.imread(filename)
        if original_img is not None:
            images.append(original_img)
        else:
            print("image not found!!!")
            break
        original_img_height, original_img_width = original_img.shape[:2]
        person_num = len(bbox_list[ith_img])

        # for each cropped and resized human image, forward it to PoseNet
        output_pose_2d_list = []
        output_pose_3d_list = []

        for n in range(person_num):
            bbox = process_bbox(np.array(bbox_list[ith_img][n]), original_img_width, original_img_height)
            img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False)
            img = transform(img).cuda()[None,:,:,:]

            # forward
            with torch.no_grad():
                pose_3d = model(img) # x,y: pixel, z: root-relative depth (mm)

            # inverse affine transform (restore the crop and resize)
            pose_3d = pose_3d[0].cpu().numpy()
            pose_3d[:,0] = pose_3d[:,0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_3d[:,1] = pose_3d[:,1] / cfg.output_shape[0] * cfg.input_shape[0]
            pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
            pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            output_pose_2d_list.append(pose_3d[:,:2].copy())

            # root-relative discretized depth -> absolute continuous depth
            pose_3d[:,2] = (pose_3d[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_depth_list[pic-1][n]
            pose_3d = pixel2cam(pose_3d, focal, princpt)
            output_pose_3d_list.append(pose_3d.copy())# ?

        if pic == 500:
            vis_img = original_img.copy()
            for n in range(person_num):
                vis_kps = np.zeros((3,joint_num))
                vis_kps[0,:] = output_pose_2d_list[n][:,0]
                vis_kps[1,:] = output_pose_2d_list[n][:,1]
                vis_kps[2,:] = 1 #kps thresh in vis_keypoints
                vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
            cv2.imwrite(img_file_name+'_'+trained_set+'.jpg', vis_img)

        # 1 dim: people, 2 dim: num of joints, 3 dim: xyz coords
        vis_kps = np.array(output_pose_3d_list)
        ''' # visualize all pictures
        if pic == 500:
            vis_3d_multiple_skeleton(pic, bbox_list[ith_img], vis_kps, np.ones_like(vis_kps), skeleton, joints_name, dataset, 'output_pose_3d (x,y,z: camera-centered. mm.)')
        '''
        ''' take coords only related to reba from prediction, size(#imgs, 1, 16 ,3), cam coord, pelvis centered'''
        kpt_estimation_all[ith_img,:,:,:] = coords_from_other_dataset(ith_img, vis_kps, dataset) #print(adjusted_kpt_3d.shape)
        ''' take coords only related to reba from vicon, size(#imgs, 1, 16 ,3), orgin is pelvis'''
        kpt_gt_all[ith_img,:,:,:] = reba_coords_from_our_model(ith_img, frames, person_num, subject) # read ground truth coords of current image
        ''' Convert world coord to cam coord for GT'''
        ''' REBA score calc for both data source'''
        # REBA score using predicted coords
        reba_score_pred[ith_img, :], reba_angles_pred[ith_img,:,:] = reba_calculator(pic, kpt_estimation_all[ith_img,:,:,:], dataset, writer)
        # REBA score using ground truth coords
        reba_score_gt[ith_img, :], reba_angles_gt[ith_img,:,:] = reba_calculator(pic, kpt_gt_all[ith_img,:,:,:], "ground_truth", writer1)
        mpjpe_per_frame = get_MPJPE(kpt_estimation_all[ith_img,:,:,:], kpt_gt_all[ith_img,:,:,:], person_num, joint_num)
        mpjpe_total += mpjpe_per_frame
        ith_img += 1
    print("MPJPE for %s: %f" % (img_file_name, (mpjpe_total/ith_img)))
    reba_angles_file.close()
    reba_vicon_file.close()
    write_coords_to_csv(kpt_estimation_all, 'prediction', csv_joints_title,img_file_name)
    write_coords_to_csv(kpt_gt_all, 'vicon', csv_joints_title,img_file_name)
    #kpt_file.close()

    '''comparison = joints_comparison(reba_joints_pred, reba_joints_vicon)'''
    #comparison = angles_comparison(reba_angles_pred, reba_angles_gt, csv_angles_title)

def dataset_info(dataset):
    # MuCo joint set, use this if model is trained with MuCo
    if dataset == 'MuCo':
        joint_num = 21
        joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
        skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

    # H36M joint set, use this if model is trained with H36M
    if dataset == 'H36M':
        joint_num = 18 # original:17, but manually added 'Thorax'
        joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
        flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )

    # CVRE3D joint set, use this if model is trained with CVRE3D
    if dataset == 'CVRE':
        joint_num = 18
        joints_name = ('Head', 'Nose', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hand', 'R_Hand', 'Pelvis', 'Thorax')
        flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15) )
        skeleton = ( (0, 1), (1, 17), (2, 17), (5, 17), (2, 3), (3, 4), (4, 14), (5, 6), (6, 7), (7, 15), (16, 17), (8, 16), (11, 16), (8, 9), (9, 10), (11, 12), (12, 13) )
    return joint_num, joints_name, flip_pairs, skeleton

def main():
    # argument parsing
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    dataset = args.dataset

    joint_num, joints_name, flip_pairs, skeleton = dataset_info(dataset)

    # snapshot load
    #model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
    model_path = args.test_epoch
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_pose_net(cfg, False, joint_num)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    # change to make sure trained model is same as used model

    model.load_state_dict(ckpt['network'])
    model.eval()
    # prepare input image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
    for i in range(0,1):
        for j in range(6,7):
            img_file_name = 'subj_%02d_act_%02d_cam_01' % (i+1, j+1)
            print("Processing %s ......" % img_file_name)
            pose_estimation(args, img_file_name, dataset, model, transform, joint_num, joints_name, flip_pairs, skeleton)

if __name__ == "__main__":
    main()
