import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
from datetime import datetime

def download_images_from_google(name, query, num_images):
    # Tạo thư mục lưu trữ hình ảnh
    if not os.path.exists(name):
        os.makedirs(name)

    # Số lượng hình ảnh cần tải
    num_images_per_request = 20

    # Số lần yêu cầu để đạt được số lượng hình ảnh mong muốn
    num_requests = (num_images + num_images_per_request - 1) // num_images_per_request

    for i in range(num_requests):
        # Vị trí bắt đầu của kết quả hình ảnh trong mỗi yêu cầu
        start_index = i * num_images_per_request

        # Tạo URL tìm kiếm trên Google và thêm tham số "tbm=isch" và "start" để chỉ tìm kiếm hình ảnh và xác định vị trí bắt đầu
        query_encoded = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={query_encoded}&tbm=isch&start={start_index}"

        # Gửi yêu cầu HTTP GET đến trang tìm kiếm Google
        response = requests.get(url)
        response.raise_for_status()

        # Phân tích HTML trang tìm kiếm Google bằng BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Tìm tất cả thẻ <img> trong trang
        image_tags = soup.find_all("img")

        # Tải về tất cả hình ảnh từ kết quả hiện tại
        for j, image_tag in enumerate(image_tags):
            image_url = image_tag["src"]
            try:
                # Tạo yêu cầu GET đến URL hình ảnh và lưu trữ dữ liệu nhận được vào file
                response = requests.get(image_url)
                response.raise_for_status()
                today = datetime.timestamp(datetime.now())
                with open(os.path.join(name, f"{name}{start_index + j + today}.jpg"), "wb") as file:
                    file.write(response.content)
                print(f"Tải về thành công hình ảnh {start_index + j + 1}")
            except requests.exceptions.RequestException as e:
                print(f"Lỗi trong quá trình tải về hình ảnh {start_index + j + 1}: {e}")

# Gọi hàm và truyền từ khóa tìm kiếm và số lượng hình ảnh cần tải
name = "songloan"
query = "Song loan"
num_images = 500
download_images_from_google(name, query, num_images)