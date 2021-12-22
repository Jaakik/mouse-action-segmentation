import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
from matplotlib import rc
import numpy as np

rc('animation', html='jshtml')

# Note: Image processing may be slow if too many frames are animated.

# Plotting constants
FRAME_WIDTH_TOP = 1024
FRAME_HEIGHT_TOP = 570

RESIDENT_COLOR = 'lawngreen'
INTRUDER_COLOR = 'skyblue'

PLOT_MOUSE_START_END = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4),
                        (3, 5), (4, 6), (5, 6), (1, 2)]

class_to_color = {'other': 'white', 'attack': 'red', 'mount': 'green',
                  'investigation': 'orange'}

# class_to_number = {s: i for i, s in enumerate(train['vocabulary'])}
#
# number_to_class = {i: s for i, s in enumerate(train['vocabulary'])}


def num_to_text(anno_list, number_to_class):
    return np.vectorize(number_to_class.get)(anno_list)


def set_figax():
    fig = plt.figure(figsize=(6, 4))

    img = np.zeros((FRAME_HEIGHT_TOP, FRAME_WIDTH_TOP, 3))

    ax = fig.add_subplot(111)
    ax.imshow(img)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax


def plot_mouse(ax, pose, color):
    # Draw each keypoint
    for j in range(7):
        ax.plot(pose[j, 0], pose[j, 1], 'o', color=color, markersize=5)

    # Draw a line for each point pair to form the shape of the mouse

    for pair in PLOT_MOUSE_START_END:
        line_to_plot = pose[pair, :]
        ax.plot(line_to_plot[:, 0], line_to_plot[
                                    :, 1], color=color, linewidth=1)


def animate_pose_sequence(video_name, keypoint_sequence, number_to_class, start_frame=0, stop_frame=100,
                          annotation_sequence=None):
    # Returns the animation of the keypoint sequence between start frame
    # and stop frame. Optionally can display annotations.
    seq = keypoint_sequence.transpose((0, 1, 3, 2))

    image_list = []

    counter = 0
    for j in range(start_frame, stop_frame):
        if counter % 20 == 0:
            print("Processing frame ", j)
        fig, ax = set_figax()
        plot_mouse(ax, seq[j, 0, :, :], color=RESIDENT_COLOR)
        plot_mouse(ax, seq[j, 1, :, :], color=INTRUDER_COLOR)

        if annotation_sequence is not None:
            annot = annotation_sequence[j]
            annot = number_to_class[annot]
            plt.text(50, -20, annot, fontsize=16,
                     bbox=dict(facecolor=class_to_color[annot], alpha=0.5))

        ax.set_title(
            video_name + '\n frame {:03d}.png'.format(j))

        ax.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(),
                                        dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))

        image_list.append(image_from_plot)

        plt.close()
        counter = counter + 1

    # Plot animation.
    fig = plt.figure()
    plt.axis('off')
    im = plt.imshow(image_list[0])

    def animate(k):
        im.set_array(image_list[k])
        return im,

    ani = animation.FuncAnimation(fig, animate, frames=len(image_list), blit=True)
    return ani


def plot_annotation_strip(annotation_sequence, start_frame=0, stop_frame=100, title="Behavior Labels"):
    # Plot annotations as a annotation strip.

    # Map annotations to a number.
    annotation_num = []
    for item in annotation_sequence[start_frame:stop_frame]:
        annotation_num.append([item])

    all_classes = list(set(annotation_sequence[start_frame:stop_frame]))

    cmap = colors.ListedColormap(['red', 'orange', 'green', 'white'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    height = 200
    arr_to_plot = np.repeat(np.array(annotation_num)[:, np.newaxis].transpose(),
                            height, axis=0)

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.imshow(arr_to_plot, interpolation='none', cmap=cmap, norm=norm)

    ax.set_yticks([])
    ax.set_xlabel('Frame Number')
    plt.title(title)

    import matplotlib.patches as mpatches

    legend_patches = []
    for item in all_classes:
        legend_patches.append(mpatches.Patch(color=class_to_color[item], label=item))

    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()