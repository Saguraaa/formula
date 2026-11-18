import unittest
from arithmetic import judge

class ArithmeticTests(unittest.TestCase):
    def test_thousands_separator(self):
        for text in ['200+800=1000', '200+800=1,000', '１，０００＋２＝１，００２',
                     '1,000.5+0.5=1,001', '1,000,000/1,000=1,000']:
            with self.subTest(text=text):
                result = judge(text)
                self.assertIs(result['is_correct'], True)
                self.assertEqual(result['raw_text'], text)
                self.assertNotIn(',', result['normalized_text'])
        self.assertIs(judge('200+800=1,001')['is_correct'], False)

    def test_malformed_commas_are_not_silently_removed(self):
        for text in ['200+800=1,00', '1,5+1=16', '12,34,567=1234567',
                     '1,,000=1000', '1,000,=1000', '1.2,000=1.2', ',100=100']:
            with self.subTest(text=text):
                result = judge(text)
                self.assertIsNone(result['is_correct'])
                self.assertEqual(result['reason'], 'unsupported_characters')
                self.assertIn(',', result['unsupported_characters'])

    def test_length_and_character_errors_are_distinct(self):
        self.assertEqual(judge('1' * 201)['reason'], 'expression_too_long')
        result = judge('200+800=1O00')
        self.assertEqual(result['reason'], 'unsupported_characters')
        self.assertEqual(result['unsupported_characters'], ['O'])

    def test_correct(self):
        for text in ['12+8=20', '0.1+0.2=0.3', '(2+3)×4=20', '8÷3=8/3', '-2+1=-1', '１２＋８＝２０']:
            with self.subTest(text=text): self.assertIs(judge(text)['is_correct'], True)
    def test_wrong(self):
        for text in ['560+70=600', '2+2=5']:
            self.assertIs(judge(text)['is_correct'], False)
    def test_uncertain(self):
        for text in ['', '1+1=', '1/0=1', '2**3=8', '__import__(1)=2', '1=1=1', '1//2=0', '2O+1=21']:
            with self.subTest(text=text): self.assertIsNone(judge(text)['is_correct'])

if __name__ == '__main__': unittest.main()
