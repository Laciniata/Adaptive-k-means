import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

from matplotlib.ticker import FixedLocator, FixedFormatter

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


def data_init():
    """
    数据初始化。

    Returns:
        data (pd.DataFrame): 数据集
    """
    data = pd.read_csv("dataset/dataset.csv")
    data.drop('编号', axis=1, inplace=True)
    return data


def select_nearest_cluster(data_array: np.ndarray, means_array: np.ndarray):
    """
    选取距离最近的簇。

    Args:
        data_array (np.ndarray): 数据
        means_array (np.ndarray): 每个簇的均值

    Returns:
        (np.ndarray): 每个样本的簇的编号，从0开始
    """
    distances = []
    for i in range(means_array.shape[0]):
        distance = np.linalg.norm(data_array - means_array[i], axis=1)
        distances.append(distance)
    distances_array = np.array(distances)
    return distances_array.argmin(axis=0)


def k_means(data: pd.DataFrame, k: int):
    """
    k均值算法。

    Args:
        data (pd.DataFrame): 数据集
        k (int): 分簇数量

    Returns:
        clusters_array (list): 每个元素是一个np.ndarray，这个矩阵由所有分到该簇的数据样本组成。
        means_array (np.ndarray): 每个簇的均值
    """
    data_array = data.to_numpy()
    means = data.sample(k, axis=0)
    means_array = means.to_numpy()
    means_changed = True
    while means_changed:
        means_changed = False
        clusters = [[] for i in range(k)]
        cluster_serial_numbers = select_nearest_cluster(data_array, means_array)
        for c, sample in zip(cluster_serial_numbers, data_array):
            clusters[c].append(sample)
        clusters_array = [np.array(c) for c in clusters]
        for ca in clusters_array:  # 处理有簇无元素的情况：递归，直到不存在该情况
            if len(ca) == 0:
                clusters_array, means_array = k_means(data, k)
                return clusters_array, means_array
        new_means_array = np.array([ca.mean(axis=0) for ca in clusters_array])
        if not np.allclose(means_array, new_means_array):
            means_changed = True
            means_array = copy.deepcopy(new_means_array)
    # print(cluster_serial_numbers)  # 调试代码
    return clusters_array, new_means_array


def cluster_plot(clusters_array: list, means_array: np.ndarray, ax):
    """
    绘制分类后的散点图，每个簇的点用不同的颜色，每个簇的均值用+标识，并在图中标有簇的编号。

    Args:
        clusters_array (list): 每个元素是一个np.ndarray，这个矩阵由所有分到该簇的数据样本组成。
        means_array (np.ndarray): 每个簇的均值
        ax (matplotlib.axes._subplots.AxesSubplot): 子图

    Returns:
        None
    """
    for cluster in clusters_array:
        ax.scatter(cluster[:, 0], cluster[:, 1], s=20)
    ax.scatter(means_array[:, 0], means_array[:, 1], marker='+', color='red', s=60)
    for i in range(means_array.shape[0]):
        ax.annotate(i, (means_array[i, 0], means_array[i, 1]))
    ax.axis([0.2, 0.8, 0, 0.5])


def plot_boundary(means_array: np.ndarray, ax, k: int, axes: list = [0.2, 0.8, 0, 0.5], contour: bool = True):
    """
    绘制分类结果轮廓图。

    Args:
        means_array (np.ndarray): 每个簇的均值
        ax (matplotlib.axes._subplots.AxesSubplot): 子图
        k (int): 分簇数
        axes (list): 坐标轴范围
        alpha (float): 不透明度
        contour (bool): 是否绘制轮廓线

    Returns:
        None
    """
    x1s = np.linspace(axes[0], axes[1], 600)
    x2s = np.linspace(axes[2], axes[3], 600)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = select_nearest_cluster(X_new, means_array)
    y_pred = np.array(y_pred).reshape(x1.shape)
    plt.contourf(x1, x2, y_pred, levels=k, alpha=0.3, cmap='jet')
    if contour:
        plt.contour(x1, x2, y_pred, levels=k, cmap='jet', alpha=0.8)
    # x1 = x1.flatten()
    # x2 = x2.flatten()
    # y_pred = y_pred.flatten()
    # for i in range(y_pred.shape[0]):
    #     ax.annotate(int(y_pred[i]), (x1[i], x2[i]))
    ax.axis(axes)
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$x_2$", fontsize=18, rotation=0)


def loop_and_save_plot(data: pd.DataFrame, k: int, loop_times: int):
    """
    多次训练k均值，并绘制分类结果。

    Args:
        data (pd.DataFrame): 数据集
        k (int): 分簇数
        loop_times (int): 循环次数

    Returns:
        None
    """
    for i in range(loop_times):
        cluster_result, mean_result = k_means(data, k)
        # fig = plt.figure(figsize=[21.33, 11.25])
        fig = plt.figure(figsize=[13.5, 11.25])
        # fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cluster_plot(cluster_result, mean_result, ax)
        plot_boundary(mean_result, ax=ax, k=k)
        plt.savefig("{}".format(i + 1), dpi=300, bbox_inches='tight')
        print('{}.png saved.'.format(i + 1))


def calinski_harabasz(cluster_array: list, means_array: np.ndarray):
    """
    计算 Calinski-Harabasz Index.

    Args:
        clusters_array (list): 每个元素是一个np.ndarray，这个矩阵由所有分到该簇的数据样本组成。
        means_array (np.ndarray): 每个簇的均值

    Returns:
        ch_score (np.float64): Calinski-Harabasz Index
    """
    within_cluster_dispersion = np.zeros((2, 2))
    # between_group_dispersion = np.zeros((2, 2))
    bkt = 0
    center_of_data = np.vstack(cluster_array).mean(axis=0)
    for i in range(len(cluster_array)):
        deviation_w = cluster_array[i] - means_array[i]
        within_cluster_dispersion += deviation_w.T @ deviation_w
        deviation_b = means_array[i] - center_of_data
        # deviation_b = deviation_b.reshape(1, 2)
        # between_group_dispersion += cluster_array[i].shape[0] * deviation_b.T @ deviation_b
        bkt += len(cluster_array[i]) * np.sum(deviation_b ** 2)
    number_of_data = sum([c_array.shape[0] for c_array in cluster_array])
    number_of_cluster = means_array.shape[0]
    # ch_score = between_group_dispersion.trace() / within_cluster_dispersion.trace() * (
    #         number_of_data - number_of_cluster) / (number_of_cluster - 1.)
    ch_score = bkt / within_cluster_dispersion.trace() * (
            number_of_data - number_of_cluster) / (number_of_cluster - 1.)
    return ch_score


def traverse_ch_and_plot(data: pd.DataFrame, k_range: list, loop_times: int, plot_cluster: bool):
    """
    循环指定次数，计算指定k范围下各k对应的CH指数，并绘制CH-k曲线图，同时还可绘制最佳分簇图。

    Args:
        data (pd.DataFrame): 数据集
        k_range (list): k的取值范围，遍历步长为1。
        loop_times (int): 循环次数
        plot_cluster (bool): 是否绘制所有k下的最佳分簇图

    Returns:
        None
    """
    ch_score_global_list = []
    for k in range(k_range[0], k_range[1]):
        print(k)
        ch_score_k_list = []
        if plot_cluster:
            best_ch = 0
            best_cluster_array = []
            best_mean_array = np.zeros((k, 2))
        for i in range(loop_times):
            cluster_result, mean_result = k_means(data, k)
            ch_score = calinski_harabasz(cluster_result, mean_result)
            ch_score_k_list.append(ch_score)
            if best_ch < ch_score:
                if plot_cluster:
                    best_cluster_array = copy.deepcopy(cluster_result)
                    best_mean_array = copy.deepcopy(mean_result)
        ch_score_global_list.append(np.mean(ch_score_k_list))
        if plot_cluster:
            fig = plt.figure(figsize=[13.5, 11.25])
            ax = fig.add_subplot(1, 1, 1)
            cluster_plot(best_cluster_array, best_mean_array, ax)
            plot_boundary(best_mean_array, ax=ax, k=k)
            plt.savefig("best clusters k = {} (CH)".format(k), dpi=300, bbox_inches='tight')
    fig = plt.figure(figsize=[13.5, 11.25])
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(k_range[0], k_range[1]), ch_score_global_list)
    plt.savefig("CH-k", dpi=300, bbox_inches='tight')
    print(ch_score_global_list)


def select_next_nearest_cluster(sample: np.ndarray, means_array: np.ndarray):
    """
    选取距离第二近的簇。

    Args:
        sample (np.ndarray): 数据
        means_array (np.ndarray): 每个簇的均值

    Returns:
        (np.ndarray): 第二近的簇的编号，从0开始
    """
    distances = []
    for i in range(means_array.shape[0]):
        distance = np.linalg.norm(sample - means_array[i], axis=0)
        distances.append(distance)
    distances_array = np.array(distances)
    return np.argpartition(distances_array, 1)[1]  # 返回第2小的元素


def silhouette_coefficient(clusters_array: list, means_array: np.ndarray):
    """
    计算 Silhouette Coefficient 轮廓系数。

    Args:
        clusters_array (list): 每个元素是一个np.ndarray，这个矩阵由所有分到该簇的数据样本组成。
        means_array (np.ndarray): 每个簇的均值

    Returns:
        s_list (list): 每个元素是一个np.ndarray，这个矩阵由所有分到该簇的样本的轮廓系数组成。
        mean_silhouette_coefficient (np.float64): 平均轮廓系数
    """
    s_list = []
    for i in range(len(clusters_array)):
        s_i = np.zeros((clusters_array[i].shape[0], 1))
        for j in range(clusters_array[i].shape[0]):
            if clusters_array[i].shape[0] == 1:
                a_j = 0
            else:
                a_j = np.sum(np.sqrt(np.sum(np.power((clusters_array[i] - clusters_array[i][j, :]), 2), axis=1))) / (
                        clusters_array[i].shape[0] - 1)
            nnc = select_next_nearest_cluster(clusters_array[i][j, :], means_array)
            b_j = np.mean(np.sqrt(np.sum(np.power((clusters_array[nnc] - clusters_array[i][j, :]), 2), axis=1)))
            s_j = (b_j - a_j) / max(b_j, a_j)
            s_i[j] = s_j
        s_list.append(s_i)
    mean_silhouette_coefficient = np.vstack(s_list).mean()
    return s_list, mean_silhouette_coefficient


def plot_silhouette_coefficient(s_list: list, mean_silhouette_coefficient: float, k: int):
    """
    绘制每个簇所有样本的轮廓系数。

    Args:
        s_list (list): 每个元素是一个np.ndarray，这个矩阵由所有分到该簇的样本的轮廓系数组成。
        mean_silhouette_coefficient (np.float64): 平均轮廓系数
        k (int): 分簇数量

    Returns:
        None
    """
    fig = plt.figure(figsize=[13.5, 11.25])
    ax = fig.add_subplot(1, 1, 1)
    sc_of_samples = np.vstack(s_list).flatten()
    interval = len(sc_of_samples) // 30  # 不同簇的绘图间隔
    plot_position = interval  # 当前簇绘图高度
    ticks = []
    for i in range(k):
        s_list[i] = s_list[i].flatten()
        coeff_array = np.sort(s_list[i])
        color = mpl.cm.get_cmap('plasma', k)
        color = color.colors[i]
        ax.fill_betweenx(np.arange(plot_position, plot_position + len(coeff_array)), 0, coeff_array,
                         facecolor=color, edgecolor='face', alpha=0.7)  # 绘制簇，并填充左侧
        ticks.append(plot_position + len(coeff_array) // 2)
        plot_position += len(coeff_array) + interval  # 当前高度上移
    ax.axvline(x=mean_silhouette_coefficient, color="blue", linestyle="--")  # 在平均轮廓系数处画一道直线
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(range(k)))
    ax.set_ylabel("簇", rotation=0)
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_title("$k={}$".format(k), fontsize=16)
    plt.savefig("Silhouette Coefficient k={}".format(k), dpi=300, bbox_inches='tight')


def traverse_silhouette_coefficient_and_plot(data: pd.DataFrame, k_range: list, loop_times: int,
                                             plot_cluster: bool):
    """
    循环指定次数，计算指定k范围下各k对应的平均轮廓系数，并绘制SC-k曲线图，同时还可绘制最佳分簇图。

    Args:
        data (pd.DataFrame): 数据集
        k_range (list): k的取值范围，遍历步长为1。
        loop_times (int): 循环次数
        plot_cluster (bool): 是否绘制所有k下的最佳分簇图

    Returns:
        None
    """
    silhouette_coefficient_value_list = []
    for k in range(k_range[0], k_range[1]):
        print(k)
        best_sc = -1
        best_s_list = []
        if plot_cluster:
            best_cluster_array = []
            best_mean_array = np.zeros((k, 2))
        for i in range(loop_times):
            cluster_array, mean_array = k_means(data, k)
            s_list, mean_silhouette_coefficient = silhouette_coefficient(cluster_array, mean_array)
            if best_sc < mean_silhouette_coefficient:
                best_sc = mean_silhouette_coefficient
                best_s_list = copy.deepcopy(s_list)
                if plot_cluster:
                    best_cluster_array = copy.deepcopy(cluster_array)
                    best_mean_array = copy.deepcopy(mean_array)
        plot_silhouette_coefficient(best_s_list, best_sc, k)
        silhouette_coefficient_value_list.append(best_sc)
        if plot_cluster:
            fig = plt.figure(figsize=[13.5, 11.25])
            ax = fig.add_subplot(1, 1, 1)
            cluster_plot(best_cluster_array, best_mean_array, ax)
            plot_boundary(best_mean_array, ax=ax, k=k)
            plt.savefig("best clusters k = {} (SC)".format(k), dpi=300, bbox_inches='tight')
            plt.close(fig)
            plt.cla()
    fig = plt.figure(figsize=[13.5, 11.25])
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(k_range[0], k_range[1]), silhouette_coefficient_value_list)
    print(silhouette_coefficient_value_list)
    plt.savefig("SC-k", dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.cla()


def select_best_k(data: pd.DataFrame, k_range: list, loop_times: int, index_type: str):
    """
    选择最佳的分簇数量。

    Args:
        data (pd.DataFrame): 数据集
        k_range (list): k的取值范围，遍历步长为1。
        loop_times (int): 循环次数
        index_type (str): 指标类型，'CH'表示Calinski Harabasz Index，'SC'表示Silhouette Coefficient。

    Returns:
        最佳的分簇数量
    """
    if index_type == 'CH':
        ch_score_global_list = []
        for k in range(k_range[0], k_range[1]):
            print(k)
            ch_score_k_list = []
            for i in range(loop_times):
                cluster_result, mean_result = k_means(data, k)
                ch_score = calinski_harabasz(cluster_result, mean_result)
                ch_score_k_list.append(ch_score)
            ch_score_global_list.append(np.mean(ch_score_k_list))
        return np.array(ch_score_global_list).argmax() + k_range[0]
    elif index_type == 'SC':
        silhouette_coefficient_value_list = []
        for k in range(k_range[0], k_range[1]):
            print(k)
            best_sc = -1
            for i in range(loop_times):
                cluster_array, mean_array = k_means(data, k)
                s_list, mean_silhouette_coefficient = silhouette_coefficient(cluster_array, mean_array)
                if best_sc < mean_silhouette_coefficient:
                    best_sc = mean_silhouette_coefficient
            silhouette_coefficient_value_list.append(best_sc)
        return np.array(silhouette_coefficient_value_list).argmax() + k_range[0]


def adaptive_k_means(data: pd.DataFrame, k_range: list, loop_times: int, index_type: str):
    """
    自适应的k均值算法。

    Args:
        data (pd.DataFrame): 数据集
        k_range (list): k的取值范围，遍历步长为1。
        loop_times (int): 循环次数
        index_type (str): 指标类型，'CH'表示Calinski Harabasz Index，'SC'表示Silhouette Coefficient。

    Returns:
        None
    """
    k = select_best_k(data, k_range, loop_times, index_type)
    print('The best k = {}'.format(k))
    best_score = -1
    best_cluster_array = None
    best_mean_array = None
    if index_type == 'CH':
        for i in range(loop_times):
            cluster_array, mean_array = k_means(data, k)
            ch_score = calinski_harabasz(cluster_array, mean_array)
            if ch_score > best_score:
                best_score = ch_score
                best_cluster_array = copy.deepcopy(cluster_array)
                best_mean_array = copy.deepcopy(mean_array)
    elif index_type == 'SC':
        for i in range(loop_times):
            cluster_array, mean_array = k_means(data, k)
            _, mean_silhouette_coefficient = silhouette_coefficient(cluster_array, mean_array)
            if best_score < mean_silhouette_coefficient:
                best_score = mean_silhouette_coefficient
                best_cluster_array = copy.deepcopy(cluster_array)
                best_mean_array = copy.deepcopy(mean_array)
    fig = plt.figure(figsize=[13.5, 11.25])
    ax = fig.add_subplot(1, 1, 1)
    cluster_plot(best_cluster_array, best_mean_array, ax)
    plot_boundary(best_mean_array, ax=ax, k=k)


if __name__ == '__main__':
    k_value = 4
    data_set = data_init()

    # cluster_result, mean_result = k_means(data_set, k_value)
    # fig = plt.figure(figsize=[13.5, 11.25])
    # ax = fig.add_subplot(1, 1, 1)
    # # ax.scatter(data_set.iloc[:, 0], data_set.iloc[:, 1], s=20)
    # cluster_plot(cluster_result, mean_result, ax)
    # plt.savefig("cluster_plot example", dpi=300, bbox_inches='tight')

    # traverse_ch_and_plot(data_set, [2, 30], 1000, True)
    # traverse_silhouette_coefficient_and_plot(data_set, [2, 31], 500, True)

    adaptive_k_means(data_set, [2, 10], 100, 'SC')
    plt.show()
