from flask import Flask, render_template, request, send_file, redirect
from PIL import Image, ImageFilter
import io
import cv2
import numpy as np
import os

app = Flask(__name__)

# def apply_roberts_operator(image):
#     # Chuyển ảnh thành mảng numpy
#     image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
#     # Áp dụng toán tử Roberts
#     roberts_x = cv2.filter2D(image_array, -1, np.array([[1, 0], [0, -1]]))
#     roberts_y = cv2.filter2D(image_array, -1, np.array([[0, 1], [-1, 0]]))
#
#     # Tính toán biểu đồ gradient tổng hợp
#     roberts_gradient = cv2.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)
#
#     # Chuyển mảng numpy trở lại ảnh PIL
#     roberts_image = Image.fromarray(cv2.cvtColor(roberts_gradient, cv2.COLOR_BGR2RGB))
#
#     return roberts_image
#
# def apply_sobel_operator(image):
#     # Chuyển ảnh thành mảng numpy
#     image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
#     # Áp dụng toán tử Sobel
#     sobel_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)
#
#     # Tính toán biểu đồ gradient tổng hợp
#     sobel_gradient = cv2.magnitude(sobel_x, sobel_y)
#
#     # Chuyển mảng numpy trở lại ảnh PIL
#     sobel_image = Image.fromarray(cv2.cvtColor(sobel_gradient.astype(np.uint8), cv2.COLOR_BGR2RGB))
#
#     return sobel_image
#
# def apply_prewitt_operator(image):
#     # Chuyển ảnh thành mảng numpy
#     image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
#     # Áp dụng toán tử Prewitt
#     prewitt_x = cv2.filter2D(image_array, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
#     prewitt_y = cv2.filter2D(image_array, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
#
#     # Tính toán biểu đồ gradient tổng hợp
#     prewitt_gradient = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
#
#     # Chuyển mảng numpy trở lại ảnh PIL
#     prewitt_image = Image.fromarray(cv2.cvtColor(prewitt_gradient, cv2.COLOR_BGR2RGB))
#
#     return prewitt_image
# laplacian
# def apply_laplacian_filter(image):
#     # Chuyển ảnh thành mảng numpy
#     image_array = np.array(image)
#
#     # Áp dụng bộ lọc Laplacian
#     laplacian_filtered = cv2.Laplacian(image_array, cv2.CV_64F)
#
#     # Tính toán biểu đồ gradient tổng hợp
#     laplacian_gradient = np.abs(laplacian_filtered)
#
#     # Chuyển mảng numpy trở lại ảnh PIL
#     laplacian_image = Image.fromarray(laplacian_gradient.astype(np.uint8))
#
#     return laplacian_image

def enhance_image_with_laplacian(image, alpha=1.5, beta=0.5):
    image=image.convert("L")
    # Chuyển ảnh thành mảng numpy
    image_array = np.array(image)

    # Áp dụng bộ lọc Laplacian
    laplacian_filtered = cv2.Laplacian(image_array, cv2.CV_64F)

    # Tăng cường ảnh bằng cộng thêm một phần của biểu đồ Laplacian
    enhanced_image_array = np.clip(image_array + alpha * laplacian_filtered + beta, 0, 255).astype(np.uint8)

    # Chuyển mảng numpy trở lại ảnh PIL
    enhanced_image = Image.fromarray(enhanced_image_array)

    return enhanced_image

# def apply_roberts_operator(image):
#     # Chuyển ảnh thành mảng numpy
#     image_array = np.array(image)
#
#     # Tạo kernel cho toán tử Roberts
#     roberts_x_kernel = np.array([[1, 0], [0, -1]],dtype=np.float32)
#     roberts_y_kernel = np.array([[0, 1], [-1, 0]],dtype=np.float32)
#
#     # Áp dụng toán tử Roberts
#     roberts_x = cv2.filter2D(image_array, -1, roberts_x_kernel)
#     roberts_y = cv2.filter2D(image_array, -1, roberts_y_kernel)
#
#     # # Tính toán biểu đồ gradient tổng hợp
#     # roberts_gradient = np.sqrt(roberts_x**2 + roberts_y**2)
#     roberts_x_edges = roberts_x.astype(np.float32)
#     roberts_y_edges = roberts_y.astype(np.float32)
#     # Chuyển mảng numpy trở lại ảnh PIL
#     processedImage = cv2.magnitude(roberts_x_edges, roberts_y_edges)
#     processedImage = processedImage.astype(np.uint8)
#     roberts_image = Image.fromarray(processedImage.astype(np.uint8))
#
#     return roberts_image
def apply_roberts_operator(img_raw):
    img = img_raw.convert("L")
    img = np.array(img)
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, -1], [1, 0]], dtype=np.float32)

    roberts_x_edges = cv2.filter2D(img, -1, roberts_x)
    roberts_y_edges = cv2.filter2D(img, -1, roberts_y)

    roberts_x_edges = roberts_x_edges.astype(np.float32)
    roberts_y_edges = roberts_y_edges.astype(np.float32)

    processedImage = cv2.magnitude(roberts_x_edges, roberts_y_edges)
    processedImage = processedImage.astype(np.uint8)

    pil_image = Image.fromarray(processedImage)
    return pil_image
def apply_sobel_operator(image):
    # Chuyển ảnh thành mảng numpy
    gray_image = np.array(image)

    # Tạo kernel cho toán tử Sobel
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    processedImage = cv2.magnitude(sobel_x, sobel_y)
    processedImage = processedImage.astype(np.uint8)

    pil_image = Image.fromarray(processedImage)
    return pil_image

def apply_prewitt_operator(image):
    image_array = np.array(image)

    # Tạo kernel cho toán tử Prewitt
    prewitt_kernel_x = cv2.getDerivKernels(1, 0, 3, normalize=True)
    prewitt_kernel_y = cv2.getDerivKernels(0, 1, 3, normalize=True)

    # Áp dụng toán tử Prewitt
    prewitt_x = cv2.filter2D(
        image_array, cv2.CV_64F, prewitt_kernel_x[0] * prewitt_kernel_x[1]
    )
    prewitt_y = cv2.filter2D(
        image_array, cv2.CV_64F, prewitt_kernel_y[0] * prewitt_kernel_y[1]
    )

    # # Tính toán biểu đồ gradient tổng hợp
    # prewitt_gradient = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
    processedImage = cv2.magnitude(prewitt_x, prewitt_y)
    # Chuyển mảng numpy trở lại ảnh PIL
    prewitt_image = Image.fromarray(processedImage.astype(np.uint8))

    return prewitt_image


#  mở ảnh
def apply_opening(image, iterations=1):
    # Chuyển ảnh thành mảng numpy
    image_array = np.array(image)

    # Áp dụng phép co ảnh (erosion)
    kernel = np.ones((5, 5), np.uint8)  # Kernel có thể được thay đổi theo kích thước mong muốn
    eroded_array = cv2.erode(image_array, kernel, iterations=iterations)

    # Áp dụng phép dãn ảnh (dilation)
    dilated_array = cv2.dilate(eroded_array, kernel, iterations=iterations)

    # Chuyển mảng numpy trở lại ảnh PIL
    opened_image = Image.fromarray(dilated_array)

    return opened_image
# đóng ảnh
def apply_closing(image, iterations=1):
    # Chuyển ảnh thành mảng numpy
    image_array = np.array(image)

    # Áp dụng phép dãn ảnh (dilation)
    kernel = np.ones((5, 5), np.uint8)  # Kernel có thể được thay đổi theo kích thước mong muốn
    dilated_array = cv2.dilate(image_array, kernel, iterations=iterations)

    # Áp dụng phép co ảnh (erosion)
    eroded_array = cv2.erode(dilated_array, kernel, iterations=iterations)

    # Chuyển mảng numpy trở lại ảnh PIL
    closed_image = Image.fromarray(eroded_array)

    return closed_image
def min_filter(image):
    # Chuyển đổi ảnh thành ảnh xám
    grayscale_image = image.convert("L")
    image_array = np.array(grayscale_image)

    # Kích thước cửa sổ
    window_size = (3, 3)

    # Tạo kernel với kích thước của cửa sổ
    kernel = np.ones(window_size, np.uint8)

    # Áp dụng phép lọc nhỏ
    min_filtered_array = cv2.erode(image_array, kernel)

    # Chuyển đổi mảng numpy trở lại ảnh
    min_filtered_image = Image.fromarray(min_filtered_array)

    return min_filtered_image

def max_filter(image):
    # Chuyển đổi ảnh thành ảnh xám
    grayscale_image = image.convert("L")
    image_array = np.array(grayscale_image)

    # Kích thước cửa sổ
    window_size = (3, 3)

    # Tạo kernel với kích thước của cửa sổ
    kernel = np.ones(window_size, np.uint8)

    # Áp dụng phép lọc lớn
    max_filtered_array = cv2.dilate(image_array, kernel)

    # Chuyển đổi mảng numpy trở lại ảnh
    max_filtered_image = Image.fromarray(max_filtered_array)

    return max_filtered_image
# Hàm xử lý lọc trung vị
def apply_median_filter(image):
    # return image.filter(ImageFilter.MedianFilter())
    grayscale_image = image.convert("L")
    image_array = np.array(grayscale_image)

    # Áp dụng bộ lọc trung vị (Median Filter)
    median_filtered_array = cv2.medianBlur(image_array, 5)

    # Chuyển đổi mảng numpy trở lại ảnh
    median_filtered_image = Image.fromarray(median_filtered_array)

    return median_filtered_image

# Hàm xử lý lọc điểm giữa
def apply_midpoint_filter(image):
    # grayscale_image = image.convert("L")
    # image_array = np.array(grayscale_image)
    # midpoint_filtered_array = cv2.medianBlur(image_array, 3)
    # midpoint_filtered_image = Image.fromarray(midpoint_filtered_array)
    # return midpoint_filtered_image
    # Chuyển đổi ảnh thành ảnh xám
    grayscale_image = image.convert("L")
    image_array = np.array(grayscale_image)

    # Áp dụng erosion và dilation với cửa sổ 5x6
    min_pixel_values = cv2.erode(image_array, None, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=255)
    max_pixel_values = cv2.dilate(image_array, None, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    # Tính giá trị trung bình (midpoint) của các giá trị pixel
    midpoint_filtered_array = (min_pixel_values + max_pixel_values) // 2

    # Chuyển đổi mảng numpy trở lại ảnh
    midpoint_filtered_image = Image.fromarray(midpoint_filtered_array)

    return midpoint_filtered_image

# phép co
def apply_erosion(image, iterations=1):
    # Chuyển ảnh thành mảng numpy
    image_array = np.array(image)

    # Áp dụng phép co ảnh (erosion) với số lần lặp được chỉ định
    kernel = np.ones((5, 5), np.uint8)  # Kernel có thể được thay đổi theo kích thước mong muốn
    eroded_array = cv2.erode(image_array, kernel, iterations=iterations)

    # Chuyển mảng numpy trở lại ảnh PIL
    eroded_image = Image.fromarray(eroded_array)

    return eroded_image

# phép dãn
def apply_dilation(image, iterations=1):
    # Chuyển ảnh thành mảng numpy
    image_array = np.array(image)

    # Áp dụng phép dãn ảnh (dilation) với số lần lặp được chỉ định
    kernel = np.ones((5, 5), np.uint8)  # Kernel có thể được thay đổi theo kích thước mong muốn
    dilated_array = cv2.dilate(image_array, kernel, iterations=iterations)

    # Chuyển mảng numpy trở lại ảnh PIL
    dilated_image = Image.fromarray(dilated_array)

    return dilated_image
# Hàm xử lý cắt ảnh
def crop_image(image):
    width, height = image.size
    left = 0
    top = 0
    right = width // 2
    bottom = height
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

# Hàm xử lý resize ảnh
def resize_image(image):
    new_width = int(image.width * 0.5)
    new_height = int(image.height * 0.5)
    resized_image = image.resize((new_width, new_height))
    return resized_image

# Hàm xử lý upload ảnh và chuyển đổi thành ảnh xám
def process_uploaded_image(file):
    image = Image.open(file)
    grayscale_image = image.convert("L")
    grayscale_image.save("static/grayscale_image.png")
    output_buffer = io.BytesIO()
    grayscale_image.save(output_buffer, format='PNG')
    output_buffer.seek(0)
    return output_buffer
# Lọc âm bản
def apply_negative_filter(image):
    # Chuyển ảnh thành mảng numpy
    image_array = np.array(image)

    # Đảo ngược giá trị pixel (làm âm bản)
    negative_image_array = 255 - image_array

    # Chuyển mảng numpy trở lại ảnh PIL
    negative_image = Image.fromarray(negative_image_array.astype(np.uint8))

    return negative_image
# Phân ngưỡng
def apply_thresholding(image, threshold=128):
    # Chuyển ảnh thành mảng numpy
    image_array = np.array(image)

    # Áp dụng phân ngưỡng
    _, thresholded_image_array = cv2.threshold(image_array, threshold, 255, cv2.THRESH_BINARY)

    # Chuyển mảng numpy trở lại ảnh PIL
    thresholded_image = Image.fromarray(thresholded_image_array.astype(np.uint8))

    return thresholded_image

# Xử lý logarit
def apply_logarithmic_transformation(image, c=1):
    # Chuyển ảnh thành mảng numpy
    image_array = np.array(image)

    # Áp dụng biến đổi logarit
    transformed_image_array = c * np.log(1 + image_array)

    # Chuẩn hóa giá trị pixel về đoạn [0, 255]
    transformed_image_array = (transformed_image_array / np.max(transformed_image_array)) * 255

    # Chuyển mảng numpy trở lại ảnh PIL
    transformed_image = Image.fromarray(transformed_image_array.astype(np.uint8))

    return transformed_image

def histogram_equalizing(img_raw):
    img_to_yuv = img_raw.convert("L")
    img_to_yuv = np.array(img_to_yuv)
    img_to_yuv= cv2.equalizeHist(img_to_yuv)


    pil_image = Image.fromarray(img_to_yuv)
    return pil_image

def average_filter (img_raw):
    img = img_raw.convert("L")
    img = np.array(img)
    m, n = img.shape

    mask = np.ones([3, 3], dtype=int)
    mask = mask / 9
    img_new = np.zeros([m, n])

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            temp = img[i - 1, j - 1] * mask[0, 0] + img[i - 1, j] * mask[0, 1] + img[i - 1, j + 1] * mask[0, 2] + img[
                i, j - 1] * mask[1, 0] + img[i, j] * mask[1, 1] + img[i, j + 1] * mask[1, 2] + img[i + 1, j - 1] * mask[
                       2, 0] + img[i + 1, j] * mask[2, 1] + img[i + 1, j + 1] * mask[2, 2]

            img_new[i, j] = temp

    img_new = img_new.astype(np.uint8)

    pil_image = Image.fromarray(img_new)
    return pil_image


def canny_operator(img_raw):
    img = np.array(img_raw)
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    canny = canny.astype(np.uint8)

    pil_image = Image.fromarray(canny)
    return pil_image


def otsu_algorithm(img_raw):
    img = np.array(img_raw)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY +
                                 cv2.THRESH_OTSU)
    thresh1 = thresh1.astype(np.uint8)

    pil_image = Image.fromarray(thresh1)
    return pil_image
# Hàm xử lý các yêu cầu từ frontend
def process_frontend_request(request):
    filter_type = request.form.get('filter_type')
    grayscale_image = Image.open("static/grayscale.png")

    if filter_type == 'median':
        processed_image = apply_median_filter(grayscale_image)
    elif filter_type == 'midpoint':
        processed_image = apply_midpoint_filter(grayscale_image)
    elif filter_type == 'crop':
        processed_image = crop_image(grayscale_image)
    elif filter_type == 'resize':
        processed_image = resize_image(grayscale_image)
    elif filter_type == 'min':
        processed_image = min_filter(grayscale_image)
    elif filter_type == 'max':
        processed_image = max_filter(grayscale_image)
    elif filter_type == 'negative':
        processed_image = apply_negative_filter(grayscale_image)
    elif filter_type == 'thresholding':
        processed_image = apply_thresholding(grayscale_image)
    elif filter_type == 'logarit':
        processed_image = apply_logarithmic_transformation(grayscale_image)
    elif filter_type == 'erosion':
        processed_image = apply_erosion(grayscale_image)
    elif filter_type == 'dilation':
        processed_image = apply_dilation(grayscale_image)
    elif filter_type == 'opening':
        processed_image = apply_opening(grayscale_image)
    elif filter_type == 'closing':
        processed_image = apply_closing(grayscale_image)
    elif filter_type == 'prewitt':
        processed_image = apply_prewitt_operator(grayscale_image)
    elif filter_type == 'laplacian':
        processed_image = enhance_image_with_laplacian(grayscale_image)
    elif filter_type == 'roberts':
        processed_image = apply_roberts_operator(grayscale_image)
    elif filter_type == 'sobel':
        processed_image = apply_sobel_operator(grayscale_image)
    elif filter_type == 'histogram':
        processed_image = histogram_equalizing(grayscale_image)
    elif filter_type == 'average':
        processed_image = average_filter(grayscale_image)
    elif filter_type == 'canny':
        processed_image = canny_operator(grayscale_image)
    elif filter_type == 'otsu':
        processed_image = otsu_algorithm(grayscale_image)
    else:
        return None

    output_buffer = io.BytesIO()
    processed_image.save(output_buffer, format='PNG')
    output_buffer.seek(0)
    return output_buffer

# Route để render trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Route để xử lý upload ảnh và chuyển đổi thành ảnh xám
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    #
    # output_buffer = process_uploaded_image(file)
    #
    # return send_file(output_buffer, mimetype='image/png', as_attachment=True, download_name='grayscale.png')
    # Lưu ảnh mới vừa upload
    file.save(os.path.join("static", "grayscale.png"))

    # Gọi hàm xử lý ảnh và lấy output buffer
    output_buffer = process_uploaded_image(file)

    return send_file(output_buffer, mimetype='image/png', as_attachment=True, download_name='grayscale.png')

# Route để xử lý các yêu cầu từ frontend
@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    output_buffer = process_frontend_request(request)

    if output_buffer is None:
        return "Invalid filter type"

    return send_file(output_buffer, mimetype='image/png', as_attachment=True, download_name='processed_image.png')

if __name__ == '__main__':
    app.run(debug=True)
