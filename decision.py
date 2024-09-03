import cv2
import numpy as np
import os
import open3d as o3d
import csv

def load_images_batch(folder, batch_size=10, scale=0.5):
    """
    Загружает изображения из указанной папки и возвращает их пакетами.
    
    Параметры:
    - folder (str): Путь к папке с изображениями.
    - batch_size (int): Количество изображений в одном пакете.
    - scale (float): Масштабный коэффициент для изменения размера изображений.
    
    Возвращает:
    - Генератор, который возвращает пакеты изображений.
    """
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        if i > 0 and i % batch_size == 0:
            yield images
            images.clear()
        
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = resize_image(img, scale)
            images.append(img)
    
    if images:
        yield images

def resize_image(img, scale=0.5):
    """
    Изменяет размер изображения на основе заданного масштабного коэффициента.
    
    Параметры:
    - img (numpy.ndarray): Исходное изображение.
    - scale (float): Масштабный коэффициент для изменения размера.
    
    Возвращает:
    - numpy.ndarray: Измененное изображение.
    """
    height, width = img.shape[:2]
    return cv2.resize(img, (int(width * scale), int(height * scale)))

def detect_and_describe(image, nfeatures=10000):
    """
    Использует SIFT для обнаружения и описания ключевых точек на изображении.
    
    Параметры:
    - image (numpy.ndarray): Исходное изображение.
    - nfeatures (int): Максимальное количество ключевых точек для обнаружения.
    
    Возвращает:
    - keypoints (list): Список ключевых точек.
    - descriptors (numpy.ndarray): Дескрипторы ключевых точек.
    """
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_keypoints_bfsearch(descriptors1, descriptors2):
    """
    Использует BFMatcher для сопоставления дескрипторов двух изображений.
    
    Параметры:
    - descriptors1 (numpy.ndarray): Дескрипторы первого изображения.
    - descriptors2 (numpy.ndarray): Дескрипторы второго изображения.
    
    Возвращает:
    - matches (list): Список сопоставленных дескрипторов.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def estimate_camera_trajectory(images, nfeatures=10000):
    """
    Оценивает траекторию камеры на основе последовательности изображений.
    
    Параметры:
    - images (list): Список изображений.
    - nfeatures (int): Максимальное количество ключевых точек для обнаружения.
    
    Возвращает:
    - trajectory (list): Список кортежей, содержащих индекс изображения, положение и ориентацию камеры.
    """
    keypoints_list = []
    descriptors_list = []
    
    for img in images:
        keypoints, descriptors = detect_and_describe(img, nfeatures)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    
    trajectory = []
    for idx, (kp1, desc1) in enumerate(zip(keypoints_list, descriptors_list)):
        if idx > 0:
            matches = match_keypoints_bfsearch(descriptors_list[idx-1], desc1)
        
        position = np.array([idx, idx, idx])  # Placeholder
        orientation = np.array([0, 0, 0])  # Placeholder
        trajectory.append((idx, position, orientation))
    
    return trajectory

def create_point_cloud(images, trajectory, chunk_size=5):
    """
    Создает облако точек на основе последовательности изображений и траектории камеры.
    
    Параметры:
    - images (list): Список изображений.
    - trajectory (list): Список кортежей, содержащих индекс изображения, положение и ориентацию камеры.
    - chunk_size (int): Количество изображений для обработки за один раз.
    
    Возвращает:
    - point_cloud (o3d.geometry.PointCloud): Облако точек.
    """
    point_cloud = o3d.geometry.PointCloud()
    
    for batch_start in range(0, len(images), chunk_size):
        batch_images = images[batch_start:batch_start + chunk_size]
        batch_trajectory = trajectory[batch_start:batch_start + chunk_size]
        
        for i, img in enumerate(batch_images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            points = cv2.goodFeaturesToTrack(gray, 500, 0.01, 10)
            if points is not None:
                points = points[:, 0, :]
                pos = batch_trajectory[i][1]
                points = np.hstack((points, np.ones((points.shape[0], 1))))
                points[:, :3] += pos
                
                point_cloud.points.extend(o3d.utility.Vector3dVector(points[:, :3]))
    
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return point_cloud

def save_point_cloud(point_cloud, filename):
    """
    Сохраняет облако точек в файл.
    
    Параметры:
    - point_cloud (o3d.geometry.PointCloud): Облако точек.
    - filename (str): Имя файла для сохранения.
    """
    o3d.io.write_point_cloud(filename, point_cloud)

def save_trajectory_to_csv(trajectory, filename):
    """
    Сохраняет траекторию камеры в CSV-файл.
    
    Параметры:
    - trajectory (list): Список кортежей, содержащих индекс изображения, положение и ориентацию камеры.
    - filename (str): Имя файла для сохранения.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Position X", "Position Y", "Position Z", "Orientation X", "Orientation Y", "Orientation Z"])
        for index, position, orientation in trajectory:
            writer.writerow([index, *position, *orientation])

def create_trajectory_line_set(trajectory):
    """
    Создает набор линий для визуализации траектории камеры.
    
    Параметры:
    - trajectory (list): Список кортежей, содержащих индекс изображения, положение и ориентацию камеры.
    
    Возвращает:
    - line_set (o3d.geometry.LineSet): Набор линий для визуализации траектории.
    """
    points = np.array([pos for idx, pos, ori in trajectory])
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    colors = [[1, 0, 0] for i in range(len(lines))]  # Цвет линии (красный)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def process_images(folder, batch_size=10, scale=0.5, nfeatures=10000):
    """
    Обрабатывает изображения из указанной папки, оценивает траекторию камеры и создает облако точек.
    
    Параметры:
    - folder (str): Путь к папке с изображениями.
    - batch_size (int): Количество изображений в одном пакете.
    - scale (float): Масштабный коэффициент для изменения размера изображений.
    - nfeatures (int): Максимальное количество ключевых точек для обнаружения.
    """
    all_images = []
    all_trajectory = []

    for batch in load_images_batch(folder, batch_size=batch_size, scale=scale):
        trajectory = estimate_camera_trajectory(batch, nfeatures)
        all_images.extend(batch)
        all_trajectory.extend(trajectory)
        del batch

    point_cloud = create_point_cloud(all_images, all_trajectory, chunk_size=batch_size)
    save_point_cloud(point_cloud, "output_bfsearch.ply")
    save_trajectory_to_csv(all_trajectory, "camera_trajectory.csv")
    
    trajectory_line_set = create_trajectory_line_set(all_trajectory)
    visualize_point_cloud_with_trajectory(point_cloud, trajectory_line_set)

def load_point_cloud_from_file(filename):
    """
    Загружает облако точек из файла.
    
    Параметры:
    - filename (str): Имя файла для загрузки.
    
    Возвращает:
    - point_cloud (o3d.geometry.PointCloud): Облако точек.
    """
    point_cloud = o3d.io.read_point_cloud(filename)
    return point_cloud

def visualize_point_cloud_with_trajectory(point_cloud, trajectory_line_set):
    """
    Визуализирует облако точек и траекторию камеры.
    
    Параметры:
    - point_cloud (o3d.geometry.PointCloud): Облако точек.
    - trajectory_line_set (o3d.geometry.LineSet): Набор линий для визуализации траектории.
    """
    o3d.visualization.draw_geometries([point_cloud, trajectory_line_set])

if __name__ == '__main__':
    process_images("TestTaskSFM/sphere_sfm/", nfeatures=10000)