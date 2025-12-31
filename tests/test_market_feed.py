
import unittest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to import run_market_feed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_market_feed import filter_instrument_keys, main

class TestMarketFeed(unittest.TestCase):
    def test_filter_instrument_keys(self):
        """Test that only keys starting with BSE_EQ|INE or NSE_EQ|INE are filtered."""
        data = {
            'instrument_key': [
                'BSE_EQ|INE123',
                'NSE_EQ|INE456',
                'BSE_EQ|IN789', # Should not match (missing E)
                'NSE_EQ|IN012', # Should not match
                'NCD_FO|14294',
                'BSE_EQ|INE_VALID',
                'NSE_EQ|INE_VALID'
            ],
            'tradingsymbol': ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        }
        df = pd.DataFrame(data)
        
        filtered_keys = filter_instrument_keys(df)
        
        expected = [
            'BSE_EQ|INE123',
            'NSE_EQ|INE456',
            'BSE_EQ|INE_VALID',
            'NSE_EQ|INE_VALID'
        ]
        
        self.assertEqual(len(filtered_keys), 4)
        self.assertEqual(filtered_keys, expected)

    def test_filter_empty(self):
        """Test filtering on data that has no matches."""
        df = pd.DataFrame({'instrument_key': ['INVALID|KEY'], 'tradingsymbol': ['X']})
        filtered_keys = filter_instrument_keys(df)
        self.assertEqual(len(filtered_keys), 0)

    def test_filter_handling_nans(self):
        """Test that NaN values in instrument_key don't crash the filter."""
        df = pd.DataFrame({
            'instrument_key': ['BSE_EQ|INE123', None, float('nan')],
            'tradingsymbol': ['A', 'B', 'C']
        })
        filtered_keys = filter_instrument_keys(df)
        self.assertEqual(filtered_keys, ['BSE_EQ|INE123'])

if __name__ == '__main__':
    unittest.main()
