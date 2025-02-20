import cv2
import numpy as np
import pyrealsense2 as rs
from imutils.perspective import four_point_transform
import imutils

# Định nghĩa khoảng giá trị HSV cho màu xanh
lower_blue = np.array([36, 106, 73])
upper_blue = np.array([161, 255, 255])

# Tạo ma trận nhị phân ứng với hướng dịch chuyển
SIGNS_LOOKUP = {
    (1, 0, 0, 1): 'Turn Right',
    (0, 0, 1, 1): 'Turn Left',
    (0, 1, 0, 0): 'Move Straight',  # Chỗ này khác với input là ảnh
    (1, 0, 1, 1): 'Turn Back',
}

# Tạo mức phân ngững
THRESHOLD = 150

def findTrafficSign(frame):
    '''Hàm tìm các đốm xanh trên khung ảnh và tìm đốm vuông lớn nhất'''
    # Lấy khung ảnh theo giá trị và tính diện tích
    frame = imutils.resize(frame, width=500)
    frameArea = frame.shape[0] * frame.shape[1]

    # Chuyển đổi ảnh màu BGR thành HSV từ frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Xác định kernel để làm mịn ảnh
    kernel = np.ones((3, 3), np.uint8)

    # Trích xuất hình ảnh nhị phân với các vùng xanh
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # morphological operations (loại bỏ nhiễu, mịn đường viền)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Tìm contours trong mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Định nghĩa biến
    detectedTrafficSign = None
    largestArea = 0
    largestRect = None

    # Nếu tìm thấy 1 contour thì tiếp tục
    if len(cnts) > 0:
        for cnt in cnts:
            # Tính toán hình chữ nhật có diện tích nhỏ nhất mà vẫn có thể bao quanh contour được chọn
            rect = cv2.minAreaRect(cnt)

            # Lấy 4 đỉnh hình chữ nhật
            box = cv2.boxPoints(rect)

            # Chuyển tọa độ của định từ số thực sang số nguyên
            box = np.int0(box)

            # Tính khoảng cách Euclidean cho mỗi cạnh của hình chữ nhật
            sideOne = np.linalg.norm(box[0] - box[1])
            sideTwo = np.linalg.norm(box[0] - box[3])

            # Tính diện tích hình chữ nhật
            area = sideOne * sideTwo
            # Tìm hình chữ nhật lớn nhất trong tất cả các contours
            if area > largestArea:
                largestArea = area
                largestRect = box

    # Vẽ đường contour của hình chữ nhật được tìm thấy lên ảnh gốc
    if largestArea > frameArea * 0.02:
        cv2.drawContours(frame, [largestRect], 0, (0, 0, 255), 2)

        # Cắt và warp vùng quan trọng
        warped = four_point_transform(mask, [largestRect][0])

        # Dùng hàm để phân biệt hướng biển báo
        detectedTrafficSign = identifyTrafficSign(warped)
        print(detectedTrafficSign)

        if detectedTrafficSign == 'Turn Right':
            print("Turn Right")
        elif detectedTrafficSign == 'Turn Left':
            print("Turn Left")
        elif detectedTrafficSign == 'Move Straight':
            print("Move Straight")
        else:
            pass

        # Tạo putText hiển thị biển báo trên ảnh
        cv2.putText(frame, detectedTrafficSign, tuple(largestRect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Hiển thị ảnh
    cv2.imshow("Original", frame)

def identifyTrafficSign(image):
    # Đảo ngược bit trong ảnh, chuyển ảnh sáng thành tối
    image = cv2.bitwise_not(image)
    # Sau đó chia kích thước ảnh và gán giá trị
    (subHeight, subWidth) = np.divide(image.shape, 10)
    subHeight = int(subHeight)
    subWidth = int(subWidth)

    # Đánh dấu các ROIs border trên ảnh
    cv2.rectangle(image, (subWidth, 4 * subHeight), (3 * subWidth, 9 * subHeight), (0, 255, 0), 2)  # Left block
    cv2.rectangle(image, (4 * subWidth, 4 * subHeight), (6 * subWidth, 9 * subHeight), (0, 255, 0), 2)  # Center block
    cv2.rectangle(image, (7 * subWidth, 4 * subHeight), (9 * subWidth, 9 * subHeight), (0, 255, 0), 2)  # Right block
    cv2.rectangle(image, (3 * subWidth, 2 * subHeight), (7 * subWidth, 4 * subHeight), (0, 255, 0), 2)  # Top block

    # Cắt 4 ROI of the sign thresh image
    leftBlock = image[4 * subHeight:9 * subHeight, subWidth:3 * subWidth]
    centerBlock = image[4 * subHeight:9 * subHeight, 4 * subWidth:6 * subWidth]
    rightBlock = image[4 * subHeight:9 * subHeight, 7 * subWidth:9 * subWidth]
    topBlock = image[2 * subHeight:4 * subHeight, 3 * subWidth:7 * subWidth]

    # Tính tỷ lệ mức độ sáng trong từng ROI, lấy tổng giá trị các pixel chia cho diện tích
    leftFraction = np.sum(leftBlock) / (leftBlock.shape[0] * leftBlock.shape[1])
    centerFraction = np.sum(centerBlock) / (centerBlock.shape[0] * centerBlock.shape[1])
    rightFraction = np.sum(rightBlock) / (rightBlock.shape[0] * rightBlock.shape[1])
    topFraction = np.sum(topBlock) / (topBlock.shape[0] * topBlock.shape[1])

    # Tạo tuple chứa 4 tỷ lệ độ sáng các vùng, kiểm tra nếu vượt qua THRESHOLD thì chuyển đổi thành và 0 nếu ngược lại
    segments = (leftFraction, centerFraction, rightFraction, topFraction)
    segments = tuple(1 if segment > THRESHOLD else 0 for segment in segments)


    # Kiểm tra segment trong SIGNS_LOOKUP
    if segments in SIGNS_LOOKUP:
        return SIGNS_LOOKUP[segments]
    else:
        return None

def detectStopSign(frame):
    # Chuyển đổi ảnh màu BGR thành HSV từ frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Phạm vi màu đỏ của stop sign
    lower_red_range1 = np.array([0, 30, 30])
    upper_red_range1 = np.array([10, 255, 255])
    lower_red_range2 = np.array([150, 30, 30])
    upper_red_range2 = np.array([190, 255, 255])

    # Xác định kernel để làm mịn ảnh
    kernel = np.ones((3, 3), np.uint8)

    # Trích xuất hình ảnh nhị phân với các vùng đỏ của stop sign
    mask1 = cv2.inRange(hsv, lower_red_range1, upper_red_range1)
    mask2 = cv2.inRange(hsv, lower_red_range2, upper_red_range2)
    mask = cv2.bitwise_or(mask1, mask2)

    # morphological operations (loại bỏ nhiễu, mịn đường viền)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Tìm contours trong mask
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Nếu tìm thấy 1 contour thì tiếp tục
    if len(cnts) > 0:
        for cnt in cnts:
            # Nếu diện tích của contour thỏa thì tiếp tục
            if cv2.contourArea(cnt) > 1500:
                # Tìm đa giác xấp xỉ contour từ đường viền ban đầu
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Nếu là hình tám cạnh thì coi là stop sign
                if len(approx) == 8:
                    print("Stop sign detected")
                    # Tạo đường bao quanh hình được contours
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Vẽ hình trên ảnh gốc từ các đường bao quanh
                    cv2.polylines(frame, [approx], True, (0, 255, 0), 5)
                    cv2.putText(frame, "Stop", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def main():
    # Initialize the Intel RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            # Convert images to numpy arrays
            frame = np.asanyarray(color_frame.get_data())

            # Process the frame
            detectStopSign(frame)
            findTrafficSign(frame)

            # Show the frame
            # cv2.imshow('RealSense', frame)

            # Stop the program if 'q' key is pressed
            if cv2.waitKey(30) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print("Dừng chương trình và đóng tất cả các cửa sổ")
                break
    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == '__main__':
    main()
