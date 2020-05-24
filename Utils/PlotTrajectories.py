import matplotlib.pyplot as plt

def plotTrajectories(trajectories):
    """
    plot a list of trajectories 画轨迹保存
    """
    for traj in trajectories:
        plt.plot(traj[:, 0].numpy(), traj[:, 1].numpy())
    # plt.savefig('trajectories.pdf')
    plt.show()