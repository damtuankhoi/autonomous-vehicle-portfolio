import cv2
import numpy as np
import matplotlib.pyplot as plt

def filter_white_color(image):
    # Chuyển đổi sang không gian màu HSV
    '''
        Trong OpenCV, giá trị Hue thường được giảm xuống phạm vi từ 0 đến 179 (hoặc từ 0 đến 255) để giảm thiểu sự phức tạp và tăng hiệu suất tính toán.
        Việc giảm phạm vi này không làm mất màu, mà chỉ làm giảm độ phân giải của kênh màu sắc, tức là số lượng giá trị có thể biểu diễn. 
        Trong thực tế, việc giảm phạm vi không ảnh hưởng nhiều đến chất lượng hình ảnh hoặc sự phát hiện màu sắc. 
        Trong OpenCV, các giá trị Hue từ 0 đến 179 tương ứng với phạm vi màu từ đỏ sang tím, với mỗi giá trị Hue biểu diễn một phần nhỏ của phổ màu sắc.
    '''
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Thiết lập ngưỡng cho màu trắng : kênh màu S (độ bão hòa màu) là rất thấp (gần 0), kênh màu V (độ sáng) là cao (gần 255), H (sắc độ) tùy ý
    lower_white = np.array([0, 77, 88])
    upper_white = np.array([225, 227, 226])

    # tạo ra một mask (mặt nạ) chứa các pixel có giá trị nằm trong khoảng màu được xác định.
    # Trả về một mảng NumPy có kích thước giống như hình ảnh ban đầu, trong đó mỗi pixel được gán giá trị 255 nếu nằm trong phạm vi màu được chỉ định và 0 nếu không.
    mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
    cv2.imshow("mask_white", mask_white)
    # mask này là ảnh xám cần chuyển về BGR để có thể AND được
    mask_white = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
    # Áp dụng mặt mask lên ảnh gốc
    result = cv2.bitwise_and(image, mask_white)
    return result

def region_of_interest(image, polygons):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    mask_image = cv2.bitwise_and(image, mask)
    return mask_image

def canny(image):
    '''
    phát hiện cạnh trong ảnh
    Args:
      image : (ndarray Shape (H, W, 3))
    Returns:
      ảnh đã được phát hiện cạnh : (ndarray Shape (H, W, 1))
    '''
    #chuyển đổi một hình ảnh màu (RGB) thành hình ảnh xám (grayscale)
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # áp dụng bộ lọc làm mờ Gaussian vào hình ảnh, giúp làm giảm nhiễu và tạo ra một hình ảnh mượt mà hơn.
    blur_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    # 50: Đây là ngưỡng dưới (lower threshold), tương tự 80: ngưỡng trên trong thuật toán Canny. 
    # Nó xác định ngưỡng dưới cho việc chấp nhận cạnh. Bất kỳ cạnh nào có độ gradient lớn hơn ngưỡng này sẽ được coi là cạnh chính thức.
    edges_img = cv2.Canny(blur_img, 50, 180) # # # # Gradient: Khoảng thay đổi giá trị độ sáng (nhỏ hay lớn)

    return edges_img

def Hougline_image(image):
    height = image.shape[0] 
    width = image.shape[1]
    threshold = int(height*2/5)

    polygons_left = np.array([[(0, threshold), (width//2, threshold), (0, height), (width//2, height)]])
    polygons_right = np.array([[ (width//2, threshold), (width, threshold), (width//2, height), (width, height)]])

    mask_left = region_of_interest(image, polygons_left)
    mask_right = region_of_interest(image, polygons_right)

    lines_left = cv2.HoughLinesP(
        mask_left, rho = 2, theta= np.pi / 180, threshold = 120, minLineLength=50, maxLineGap=10
    )
    lines_right = cv2.HoughLinesP(
        mask_right, rho = 2, theta= np.pi / 180, threshold = 120, minLineLength=50, maxLineGap=10
    )

    return (lines_left, lines_right)

def make_coordinates(image, line_parameters):
    ''''
    Xác định giao điểm của 1 đường thẳng bất kỳ với 2 đường y = H (chiều cao ảnh) và y = 3/5 *H (giá trị tự chọn)
    Args:
      image : (ndarray Shape (H, W, 3))
      line_parameters : (list), chứa slope và intercept của 1 đường thẳng
    Returns:
      tọa độ các giao điểm : (list), chứa tọa độ 2 giao điểm
    '''
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 4/5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


lane = "None"


def average_slope_intercept(image, list_lines):
    ''''
    Gom cụm các đường thẳng lại thành 1 đường thẳng duy nhất
    Args:
      image : (ndarray Shape (H, W, 3))
      lines : (ndarray shape(m, 1, 4)), m: số lines được phát hiện, mỗi line được biểu diễn bởi tọa độ 2 điểm đầu, cuối [[x_start, y_start, x_end, y_end]] 
    Returns:
      Giao điểm của đường thẳng sau khi gom cụm và  đường y = H (chiều cao ảnh) và y = 3/5 *H: (list), [x_start, y_start, x_end, y_end]
    '''
    left_fit = []
    right_fit = []

    left_line = []
    right_line = []

    global lane
    lane = "full"

    if list_lines[0] is None:
        lane = "right"
    else:
        for line in list_lines[0]:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1) # Xác định tham số của hàm bậc 1 đi qua 2 điểm.
            slope = parameters[0]
            intercept = parameters[1] 
        
            left_line.append(line)
            left_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0) # [slop_avg, intercept_avg]
        #print(left_fit_average, "left_avg")

    if list_lines[1] is None:
        if lane == "right":
            lane = "None"
            return None
        else: lane = "left"
        return (make_coordinates(image, left_fit_average),)
    
   
    for line in list_lines[1]:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) 
        slope = parameters[0]
        intercept = parameters[1] 
        right_line.append(line)
        right_fit.append((slope, intercept))
    
    right_fit_average = np.average(right_fit, axis=0)
    #print(right_fit_average, "right_avg")

    if lane == "right":
        return (make_coordinates(image, right_fit_average),)
    
    return  (make_coordinates(image, left_fit_average), make_coordinates(image, right_fit_average))

prev_center = None

def identify_center(frame, averaged_lines):
    center = None
    global prev_center
    #print(lane)
    if lane == "None":
        return prev_center
    
    if lane == "full":
        lane_center = (averaged_lines[0][2] + averaged_lines[1][2]) // 2
    elif lane == "right":
        lane_center = averaged_lines[0][2] - 90
    elif lane == "left":
        lane_center = averaged_lines[0][2] + 90

    if lane_center < 0: lane_center = 0
    bottom_y = frame.shape[0]
    center = (lane_center, int(bottom_y * (3 / 5)))
    prev_center = center
    return center

def display_lines(image, lines, color, thick):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            print(x1, y1, x2, y2, "lines")
            cv2.line(line_image, (x1, y1), (x2, y2), color, thick)
    else:
        pass
    return line_image


def display_center(image, center):
    center_image = np.copy(image)
    if center is not None:
        cv2.circle(center_image, center, 10, (0, 255, 0), -1)
    return center_image

def calculate_steering_angle(lane_center, frame_center):
    # Quy ước: Nếu góc lớn > 0 thì quẹo trái, nếu góc < 0 thì quẹo phải
    steering_angle = np.arctan(1.0 * (frame_center[0] - lane_center[0]) / (frame_center[1] - lane_center[1])) * 180 / np.pi
    # before = [-60, 60]
    # after = [60, 180]
    return steering_angle + 120


def main():
    camera = cv2.VideoCapture('FinalProject/rs_camera/ColorImg/visualize.mp4')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    camera.set(cv2.CAP_PROP_FPS, 15)
    while camera.isOpened():
        _, frame = camera.read()

        white_filtered_image = filter_white_color(frame)
        canny_image = canny(white_filtered_image)

        list_lines = Hougline_image(canny_image)
        averaged_lines = average_slope_intercept(frame, list_lines)

        line_image = display_lines(frame, averaged_lines, (0, 0, 255), 10)
        
        lane_center = identify_center(frame, averaged_lines)
        center_image = display_center(frame, lane_center)
        H, W, _ = frame.shape
        frame_center = (W//2, H)

        # Xác định góc lái
        if lane_center is not None:
            steering_angle  = calculate_steering_angle(lane_center, frame_center)
            steering_angle = round(steering_angle, 2) # Làm tròn lên 2 chữ số thập phân
            print('lane center: ',lane_center, 'frame center: ', frame_center)
            print('steering angle: ',steering_angle)

        result = cv2.addWeighted(line_image, 0.8, center_image, 1, 1)
        cv2.line(result, frame_center, (W//2, 0), (0, 255, 0), 2)
        cv2.line(result, frame_center, lane_center, (0, 0, 255), 2)
        # In ra lane (Left, Right, Full, None)
        cv2.putText(result, str(lane), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) 
        # In ra góc lái
        cv2.putText(result, str(steering_angle), (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) 
        # In ra hoành độ của Lane center.
        cv2.putText(result, str(lane_center[0]), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("result", result)
        k = cv2.waitKey(20)
        if k == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()