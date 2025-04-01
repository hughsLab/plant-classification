import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_driver_path = r'D:\PLANT\chromedriver.exe'
output_dir = r'D:\PLANT\Abeliophyllum_distichum_Nakai_Abeliophyllum_Oleaceae'

os.makedirs(output_dir, exist_ok=True)

chrome_options = Options()
# chrome_options.add_argument("--headless")  # Uncomment if you want to see the browser
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    driver.get("https://www.google.com/imghp")
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys("Red Hat")
    search_box.send_keys(Keys.RETURN)

    # Scroll down to load more images
    for _ in range(5):
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)

    image_elements = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
    print(f"Found {len(image_elements)} images on the page.")

    img_count = 0
    for img in image_elements:
        try:
            # Scroll the image into view
            driver.execute_script("arguments[0].scrollIntoView(true);", img)
            time.sleep(0.5)

            # Click on the image to get the high-resolution version
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "img.Q4LuWd")))
            img.click()
            time.sleep(1)  # Wait for high-res image to load
            
            # Get the high-resolution image URL
            high_res_image = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img.n3VNCb"))
            )
            img_url = high_res_image.get_attribute("src")

            if img_url and img_url.startswith("http"):
                print(f"Downloading image {img_count} from {img_url}")
                img_data = requests.get(img_url).content
                img_path = os.path.join(output_dir, f"Abeliophyllum_{img_count}.jpg")
                with open(img_path, "wb") as handler:
                    handler.write(img_data)

                img_count += 1
                if img_count >= 10:  # Limit to 10 images for testing
                    break
        except Exception as e:
            print(f"Failed to download image {img_count}: {e}")

finally:
    driver.quit()
