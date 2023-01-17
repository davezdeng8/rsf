import numpy as np
import mayavi.mlab as mlab
import rsf_utils
import matplotlib.pyplot as plt
import torch


def vis_flow(pc1, pc2=None, sf=None, sf_color = None, color=None, seg=False,
             color_max = None, flow_max = None, savefig = False, filename = 'flow_vis.png', view = None):
    mlab.options.offscreen = savefig
    fig = mlab.figure(bgcolor=(0,0,0), size=(1000,750))
    mlab.quiver3d(0,0,0,1,0,0, figure = fig, color=(1,0,0))
    mlab.quiver3d(0, 0, 0, 0, 1, 0, figure=fig, color=(0, 1, 0))
    mlab.quiver3d(0, 0, 0, 0, 0, 1, figure=fig, color=(0, 0, 1))

    x1 = pc1[:, 0]  # x position of point
    y1 = pc1[:, 1]  # y position of point
    z1 = pc1[:, 2]  # z position of point

    if seg and sf is not None:
        labels = rsf_utils.flow_segmentation(pc1, sf)
        mlab.points3d(x1, y1, z1, labels+1, mode="point", colormap = "rainbow", figure = fig, vmin = 0)
        mlab.show()
        return

    if color is not None:
        mlab.points3d(x1, y1, z1, color, mode="point", colormap = "rainbow", figure=fig, vmin = 0, vmax = color_max)
        mlab.colorbar(orientation='vertical')
    else:
        mlab.points3d(x1, y1, z1, mode="point", color=(1, 0, 0), figure=fig)

    if pc2 is not None:
        x2 = pc2[:, 0]  # x position of point
        y2 = pc2[:, 1]  # y position of point
        z2 = pc2[:, 2]  # z position of point
        mlab.points3d(x2, y2, z2, mode="point", color=(0, 1, 0), figure=fig)

    if sf is not None:
        flowx = sf[:, 0]
        flowy = sf[:, 1]
        flowz = sf[:, 2]
        if flow_max:
            sf_color = np.clip(sf_color, 0, flow_max)
            vecs = mlab.quiver3d(x1, y1, z1, flowx, flowy, flowz, scalars = sf_color, scale_factor = 1, figure=fig, colormap = 'rainbow', vmin=0, vmax=flow_max)
        else:
            vecs = mlab.quiver3d(x1, y1, z1, flowx, flowy, flowz, scalars = sf_color, scale_factor = 1, figure=fig, colormap = 'rainbow', vmin=0)
        if sf_color is not None:
            vecs.glyph.color_mode = 'color_by_scalar'
    if view is not None:
        mlab.view(*view)
    if savefig:
        mlab.savefig(filename)
    else:
        mlab.show()
    return

def vis_seg(pc):
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(700,700))
    mlab.quiver3d(0,0,0,1,0,0, figure = fig, color=(1,0,0))
    mlab.quiver3d(0, 0, 0, 0, 1, 0, figure=fig, color=(0, 1, 0))
    mlab.quiver3d(0, 0, 0, 0, 0, 1, figure=fig, color=(0, 0, 1))

    x = pc[:, 0]  # x position of point
    y = pc[:, 1]  # y position of point
    z = pc[:, 2]  # z position of point

    labels = rsf_utils.graph_segmentation(pc, threshold=.25)
    mlab.points3d(x, y, z, labels + 1, mode="point", colormap="rainbow", figure=fig, vmin=0)
    mlab.show()
    return

def vis_boxes_from_params(pc1, pc2, ego_transform, boxes, box_transform):
    pc1_ego = ego_transform.transform_points(pc1)
    if boxes is None:
        vis_flow(pc1_ego.detach().cpu().numpy(), pc2.detach().cpu().numpy())
        return
    print('Confidences: ')
    print(boxes[:, 0].detach().cpu().numpy())
    ego_transform_r = ego_transform.stack(*([ego_transform]*(boxes.shape[0]-1)))
    transformed_boxes = rsf_utils.transform_boxes(boxes, ego_transform_r)
    vis_transform = ego_transform_r.inverse().compose(box_transform)
    vis_boxes(pc1_ego.detach().cpu().numpy(), transformed_boxes.detach().cpu().numpy(), vis_transform, pc2.detach().cpu().numpy())

def vis_boxes(points, boxes, transform = None, pc2 = None, savefig = False, filename = 'box_vis.png', view=None, color = None):
    mlab.options.offscreen = savefig
    num_boxes = len(boxes)
    params = boxes.shape[1]
    if params!=7 and params != 8:
        print("invalid number of box parameters")
        return
    x1 = points[:, 0]  # x position of point
    y1 = points[:, 1]  # y position of point
    z1 = points[:, 2]  # z position of point
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1000, 750))
    mlab.quiver3d(0,0,0,1,0,0, figure = fig, color=(1,0,0))
    mlab.quiver3d(0, 0, 0, 0, 1, 0, figure=fig, color=(0, 1, 0))
    mlab.quiver3d(0, 0, 0, 0, 0, 1, figure=fig, color=(0, 0, 1))

    for n, box in enumerate(boxes):
        if params == 7:
            x, y, z, width, length, height, heading = box
        elif params == 8:
            c, x, y, z, width, length, height, heading = box

        center = np.array([x,y,z])
        forward = np.array([-np.sin(heading), np.cos(heading), 0])*length/2
        right = np.array([np.cos(heading), np.sin(heading), 0])*width/2
        up = np.array([0,0,height/2])
        p1 = center+forward+right+up
        p2 = center+forward+right-up
        p3 = center+forward-right+up
        p4 = center+forward-right-up
        p5 = center-forward+right+up
        p6 = center-forward+right-up
        p7 = center-forward-right+up
        p8 = center-forward-right-up
        corners = np.vstack((p1,p2,p3,p4,p5,p6,p7,p8))
        if color is None:
            use_color = (n+2)/(num_boxes+1)
        else:
            use_color = color[n]
        if params == 7:
            draw_box(corners, fig, use_color, use_color)
        elif params == 8:
            draw_box(corners, fig, use_color, c)

        if transform is not None:
            new_corners = transform[n].transform_points(torch.tensor(corners, device='cuda', dtype=torch.float32)).cpu().numpy()
            draw_box(new_corners, fig, (n+2)/(num_boxes+1), (n+2)/(num_boxes+1))

    mlab.points3d(x1, y1, z1, mode="point", color=(1, 0, 0), figure = fig)

    if pc2 is not None:
        x2 = pc2[:, 0]  # x position of point
        y2 = pc2[:, 1]  # y position of point
        z2 = pc2[:, 2]  # z position of point
        mlab.points3d(x2, y2, z2, mode="point", color=(0, 1, 0), figure=fig)

    if view is not None:
        mlab.view(*view)
    if savefig:
        mlab.savefig(filename)
    else:
        mlab.show()
    return

def draw_box(corners, figure, color, heading_color):
    center = np.mean(corners, axis = 0)
    heading_line = np.vstack((center, np.mean(corners[:4], axis = 0)))
    heading_line_plot = mlab.plot3d(heading_line[:, 0], heading_line[:, 1], heading_line[:, 2], [heading_color, heading_color], colormap="rainbow",
                figure=figure, vmin=0, vmax=1)
    p1, p2, p3, p4, p5, p6, p7, p8 = corners
    box = np.vstack([p1,p2,p4,p3,p1,p5,p6,p8,p7,p3,p4,p8,p7,p5,p6,p2])
    box_plot = mlab.plot3d(box[:, 0], box[:, 1], box[:, 2], color * np.ones(len(box)), colormap="rainbow", figure=figure, vmin=0, vmax = 1)
