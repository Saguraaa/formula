import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image
from arithmetic import judge
from web_app import create_app


class FakeEngine:
    def predict(self, path, conf, imgsz):
        return {'width': 80, 'height': 60, 'formulas': [
            dict(judge(text), id=i, bbox=[0, (i-1)*20, 80, i*20],
                 detection_confidence=.95, ocr_confidence=.9, segments=[])
            for i, text in enumerate(['1+1=2', '2+2=5', '1+1='], 1)]}


class WebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(engine=FakeEngine(), output_dir=Path(self.tmp.name))
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_upload_returns_every_crop_and_grade(self):
        data = io.BytesIO()
        Image.new('RGB', (80, 60), 'white').save(data, 'PNG')
        data.seek(0)
        response = self.client.post('/api/predict', data={'image': (data, '算式.png')})
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result['counts'], dict(total=3, correct=1, incorrect=1, uncertain=1))
        self.assertEqual([x['raw_text'] for x in result['formulas']], ['1+1=2', '2+2=5', '1+1='])
        for row in result['formulas']:
            with self.client.get(row['crop_url']) as crop:
                self.assertEqual(crop.status_code, 200)
                with Image.open(io.BytesIO(crop.data)) as image:
                    self.assertEqual(image.size, (80, 20))
        for key in ('annotated_url', 'download_url'):
            with self.client.get(result[key]) as response:
                self.assertEqual(response.status_code, 200)

    def test_invalid_upload(self):
        self.assertEqual(self.client.post('/api/predict').status_code, 400)
        response = self.client.post('/api/predict', data={'image': (io.BytesIO(b'no'), 'bad.png')})
        self.assertEqual(response.status_code, 400)

    def test_home_and_health(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/api/health').status_code, 200)

    def test_cloud_page_discloses_server_processing(self):
        with patch.dict('os.environ', {'FORMULA_DEPLOYMENT': 'cloud'}):
            text = self.client.get('/').get_data(as_text=True)
            self.assertIn('图片上传至此应用处理', text)
            self.assertNotIn('图片留在本机', text)

    def test_no_detections(self):
        self.app.config['ENGINE'].predict = lambda *args, **kwargs: {'width': 80, 'height': 60, 'formulas': []}
        data = io.BytesIO()
        Image.new('RGB', (80, 60)).save(data, 'PNG')
        data.seek(0)
        result = self.client.post('/api/predict', data={'image': (data, 'empty.png')}).get_json()
        self.assertEqual(result['counts']['total'], 0)
        self.assertEqual(result['formulas'], [])


if __name__ == '__main__':
    unittest.main()
