from handler import BaseHandler


class CustomHandler(BaseHandler):
    def __init__(self, cfg_path):
        super().__init__(cfg_path)

    # implement this to do your experiments
    def process_image(self, image):
        return image


if __name__ == "__main__":
    config_path = './config/config.yaml'
    handler = CustomHandler(config_path)
    handler.run()
