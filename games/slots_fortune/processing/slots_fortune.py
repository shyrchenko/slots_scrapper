import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import NoSuchElementException
from skimage.io import imsave

from frame_processing import process_frame
from frame_processing.symbols_identification.symbols_identifier import CorrSymbolIdentifier, SymbolsProcessor
from frame_processing.symbols_images_extraction.symbols_images_extractor import SymbolsImagesExtractor
from utils.data_models import SymbolsGrid, Vector, ROI
from utils.image_processing import crop_image
from utils.io import read_expected_symbols
from utils.logger import logger
from utils.state import set_debug_dir, is_debug_mode_activated, get_debug_dir


class LiveProcessor:
    NUMBER_OF_RETRIES = 5
    LINK = "https://slotscity.ua/ru/game/gamebeatslotscitybranded-slotscity-fortune"

    def __init__(
        self,
        symbols_images_extractor: SymbolsImagesExtractor,
        symbols_processor: SymbolsProcessor,
        grid: SymbolsGrid,
        roi: ROI
    ):
        self._symbols_images_extractor = symbols_images_extractor
        self._symbols_processor = symbols_processor
        self._grid = grid
        self._roi = roi

        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")

        self._driver = webdriver.Chrome(options=chrome_options)
        self._refresh_button: Optional[WebElement] = None

        number_of_retries = 0
        while number_of_retries < self.NUMBER_OF_RETRIES:
            try:
                self._set_up()
                break
            except NoSuchElementException:
                number_of_retries += 1


    def _set_up(self):
        self._driver.get(self.LINK)

        button = self._driver.find_element(By.XPATH, "//button[text()=' Демо игра']")
        button.click()

        time.sleep(10)

        frame = self._driver.find_element(By.ID, "game-iframe")
        self._driver.switch_to.frame(frame)
        time.sleep(1)

        self._driver.switch_to.frame(0)

        button = self._driver.find_element(By.XPATH, "//button[@id='action-start-game']")
        button.click()
        time.sleep(1)

        self._refresh_button = self._driver.find_element(By.XPATH, "//button[@id='actions-spin']")

    def _is_frame_valid(self):
        frame_is_valid = True
        try:
            self._driver.find_element(By.CLASS_NAME, 'spin__button--stop')
            frame_is_valid = False
        except NoSuchElementException:
            pass
        logger.debug(f'Frame is valid: {frame_is_valid}')
        return frame_is_valid

    def process_frames(self, interval: int) -> Dict[str, Optional[dict]]:
        if is_debug_mode_activated():
            execution_id = str(uuid.uuid4())
            set_debug_dir(Path(f'./debug/{execution_id}'))

        s = time.time()

        result = {}
        frames = 0
        while time.time() - s < interval:
            detected = False
            not_detected = 0
            while not detected:
                frame_name = f'frame_{frames}'
                if not_detected > 3:
                    break
                if not self._is_frame_valid():
                    continue

                loading_start = time.time()
                im_bytes = BytesIO(self._driver.get_screenshot_as_png())
                frame_image = Image.open(im_bytes)
                frame_image = np.array(frame_image)
                logger.debug(f'Time of loading image: {time.time() - loading_start}')

                frame_image = crop_image(frame_image, self._roi)

                result_per_frame = process_frame(
                    frame_image=frame_image,
                    frame_name=frame_name,
                    symbols_extractor=self._symbols_images_extractor,
                    reels_processor=self._symbols_processor,
                    grid=self._grid
                )
                if result_per_frame is None:
                    not_detected += 1
                else:
                    detected = True
                result[frame_name] = result_per_frame

                if is_debug_mode_activated():
                    imsave(get_debug_dir() / f"{frame_name}.png", frame_image)

                frames += 1
            self._refresh_button.click()
        return result


if __name__ == '__main__':
    import json

    ex = SymbolsImagesExtractor()
    identifier = CorrSymbolIdentifier(
        read_expected_symbols(Path(r'../symbols'))
    )
    processor = SymbolsProcessor(identifier)
    grid = SymbolsGrid(
        start_point=Vector(x=45, y=85),
        symbol_size=Vector(x=185, y=140),
        offset=Vector(x=0, y=20),
        number_of_elements=Vector(x=5, y=3)
    )
    roi = ROI(450, 1500, 129, 729)

    e = LiveProcessor(
        symbols_images_extractor=ex,
        symbols_processor=processor,
        roi=roi,
        grid=grid
    )
    r = e.process_frames(60)

    with open(get_debug_dir() / 'result.json', 'w') as f:
        json.dump(r, f)
