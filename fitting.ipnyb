import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import talib
from scipy.signal import argrelextrema
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os
import logging
import pywt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from skopt import gp_minimize
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('financial_model')

@dataclass
class Pattern:
    """
    Represents a detected chart pattern with metadata.
    
    Attributes:
        name: Name of the pattern (e.g., "Double Top", "Head and Shoulders")
        start_idx: Starting index of the pattern in the price data
        end_idx: Ending index of the pattern in the price data
        confidence: Confidence score between 0 and 1
        pattern_type: Type of pattern ("reversal" or "continuation")
        direction: Expected price direction ("bullish", "bearish", or "neutral")
        statistics: Optional dictionary containing statistical validation metrics
    """
    name: str
    start_idx: int
    end_idx: int
    confidence: float
    pattern_type: str  # 'reversal' or 'continuation'
    direction: str  # 'bullish' or 'bearish' or 'neutral'
    statistics: Dict[str, Any] = None

class PatternRecognizer:
    """
    Detects technical chart patterns in financial price data.
    
    Attributes:
        prices: Array of price data points
        volumes: Array of volume data corresponding to prices
        window: Window size for local extrema detection
        tolerance: Tolerance for pattern matching
        
    Methods:
        find_local_extrema: Identifies local maxima and minima in price data
        detect_double_top: Detects double top reversal patterns
        detect_head_and_shoulders: Detects head and shoulders reversal patterns
        detect_triangle: Detects various triangle patterns
    """
    def __init__(self, prices: np.ndarray, volumes: np.ndarray):
        self.prices = prices
        self.volumes = volumes
        self.window = 20  # Default window size for pattern recognition
        self.tolerance = 0.02  # Default tolerance for pattern similarity
        
    def find_local_extrema(self, window=None):
        """
        Finds local maxima and minima in price data.
        
        Args:
            window: Window size for extrema detection. Uses instance value if None.
            
        Returns:
            Tuple of arrays containing indices of maxima and minima
        """
        if window is None:
            window = self.window
            
        maxima = argrelextrema(self.prices, np.greater, order=window)[0]
        minima = argrelextrema(self.prices, np.less, order=window)[0]
        return maxima, minima
    
    def wavelet_pattern_detection(self, level=4):
        """
        Performs wavelet decomposition for multi-scale pattern detection.
        
        Args:
            level: The level of wavelet decomposition
            
        Returns:
            List of detected patterns at different scales
        """
        # Make sure we have enough data points for wavelet transform
        if len(self.prices) < 2**level:
            return []
            
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(self.prices, 'db4', level=level)
        
        patterns = []
        for i, coeff in enumerate(coeffs):
            # Skip if coefficient array is too small
            if len(coeff) < 2 * self.window:
                continue
                
            # Find extrema at this scale
            maxima = argrelextrema(coeff, np.greater, order=self.window//2)[0]
            minima = argrelextrema(coeff, np.less, order=self.window//2)[0]
            
            # Analyze for double tops at this scale
            if len(maxima) >= 2:
                for j in range(len(maxima)-1):
                    price1 = coeff[maxima[j]]
                    price2 = coeff[maxima[j+1]]
                    
                    if abs(price1 - price2) / max(abs(price1), 1e-10) < self.tolerance:
                        # Map the pattern back to original price index
                        scale_factor = 2**(i+1)
                        orig_start_idx = int(maxima[j] * scale_factor)
                        orig_end_idx = int(maxima[j+1] * scale_factor)
                        
                        # Clamp to valid indices in original price array
                        orig_start_idx = min(orig_start_idx, len(self.prices)-1)
                        orig_end_idx = min(orig_end_idx, len(self.prices)-1)
                        
                        # Create pattern
                        patterns.append(Pattern(
                            name=f"Wavelet Double Top (Level {i})",
                            start_idx=orig_start_idx,
                            end_idx=orig_end_idx,
                            confidence=0.7 - (0.1 * i),  # Reduce confidence for higher levels
                            pattern_type="reversal",
                            direction="bearish"
                        ))
            
            # Look for head and shoulders
            if len(maxima) >= 3:
                for j in range(len(maxima)-2):
                    left = coeff[maxima[j]]
                    head = coeff[maxima[j+1]]
                    right = coeff[maxima[j+2]]
                    
                    if (head > left and head > right and 
                        abs(left - right) / max(abs(left), 1e-10) < self.tolerance):
                        
                        # Map the pattern back to original price index
                        scale_factor = 2**(i+1)
                        orig_start_idx = int(maxima[j] * scale_factor)
                        orig_end_idx = int(maxima[j+2] * scale_factor)
                        
                        # Clamp to valid indices in original price array
                        orig_start_idx = min(orig_start_idx, len(self.prices)-1)
                        orig_end_idx = min(orig_end_idx, len(self.prices)-1)
                        
                        # Create pattern
                        patterns.append(Pattern(
                            name=f"Wavelet Head and Shoulders (Level {i})",
                            start_idx=orig_start_idx,
                            end_idx=orig_end_idx,
                            confidence=0.75 - (0.1 * i),  # Reduce confidence for higher levels
                            pattern_type="reversal",
                            direction="bearish"
                        ))
        
        return patterns
    
    def detect_double_top(self) -> Optional[Pattern]:
        """
        Detects double top reversal patterns.
        
        Returns:
            Pattern object if detected, None otherwise
        """
        maxima, _ = self.find_local_extrema()
        for i in range(len(maxima)-1):
            price1 = self.prices[maxima[i]]
            price2 = self.prices[maxima[i+1]]
            
            if abs(price1 - price2) / price1 < self.tolerance:
                neckline = min(self.prices[maxima[i]:maxima[i+1]])
                if self.prices[maxima[i+1]+1:min(maxima[i+1]+5, len(self.prices))].mean() < neckline:
                    return Pattern(
                        name="Double Top",
                        start_idx=maxima[i],
                        end_idx=maxima[i+1],
                        confidence=0.8,
                        pattern_type="reversal",
                        direction="bearish"
                    )
        return None
    
    def detect_double_bottom(self) -> Optional[Pattern]:
        """
        Detects double bottom reversal patterns.
        
        Returns:
            Pattern object if detected, None otherwise
        """
        _, minima = self.find_local_extrema()
        for i in range(len(minima)-1):
            price1 = self.prices[minima[i]]
            price2 = self.prices[minima[i+1]]
            
            if abs(price1 - price2) / max(abs(price1), 1e-10) < self.tolerance:
                neckline = max(self.prices[minima[i]:minima[i+1]])
                if len(self.prices) > minima[i+1]+5 and self.prices[minima[i+1]+1:minima[i+1]+5].mean() > neckline:
                    return Pattern(
                        name="Double Bottom",
                        start_idx=minima[i],
                        end_idx=minima[i+1],
                        confidence=0.8,
                        pattern_type="reversal",
                        direction="bullish"
                    )
        return None
    
    def detect_head_and_shoulders(self) -> Optional[Pattern]:
        """
        Detects head and shoulders reversal patterns.
        
        Returns:
            Pattern object if detected, None otherwise
        """
        maxima, minima = self.find_local_extrema()
        for i in range(len(maxima)-2):
            left = self.prices[maxima[i]]
            head = self.prices[maxima[i+1]]
            right = self.prices[maxima[i+2]]
            
            if (head > left and head > right and 
                abs(left - right) / max(abs(left), 1e-10) < self.tolerance):
                neckline = min(self.prices[maxima[i]:maxima[i+2]])
                if len(self.prices) > maxima[i+2]+5 and self.prices[maxima[i+2]+1:maxima[i+2]+5].mean() < neckline:
                    return Pattern(
                        name="Head and Shoulders",
                        start_idx=maxima[i],
                        end_idx=maxima[i+2],
                        confidence=0.85,
                        pattern_type="reversal",
                        direction="bearish"
                    )
        return None
    
    def detect_inverse_head_and_shoulders(self) -> Optional[Pattern]:
        """
        Detects inverse head and shoulders reversal patterns.
        
        Returns:
            Pattern object if detected, None otherwise
        """
        _, minima = self.find_local_extrema()
        for i in range(len(minima)-2):
            left = self.prices[minima[i]]
            head = self.prices[minima[i+1]]
            right = self.prices[minima[i+2]]
            
            if (head < left and head < right and 
                abs(left - right) / max(abs(left), 1e-10) < self.tolerance):
                neckline = max(self.prices[minima[i]:minima[i+2]])
                if len(self.prices) > minima[i+2]+5 and self.prices[minima[i+2]+1:minima[i+2]+5].mean() > neckline:
                    return Pattern(
                        name="Inverse Head and Shoulders",
                        start_idx=minima[i],
                        end_idx=minima[i+2],
                        confidence=0.85,
                        pattern_type="reversal",
                        direction="bullish"
                    )
        return None
    
    def detect_triangle(self) -> Optional[Pattern]:
        """
        Detects various triangle patterns.
        
        Returns:
            Pattern object if detected, None otherwise
        """
        maxima, minima = self.find_local_extrema(window=10)
        
        if len(maxima) < 3 or len(minima) < 3:
            return None
            
        upper_slope = np.polyfit(maxima[-3:], self.prices[maxima[-3:]], 1)[0]
        lower_slope = np.polyfit(minima[-3:], self.prices[minima[-3:]], 1)[0]
        
        if abs(upper_slope) < 0.001 and lower_slope > 0:
            return Pattern(
                name="Ascending Triangle",
                start_idx=min(minima[-3], maxima[-3]),
                end_idx=max(minima[-1], maxima[-1]),
                confidence=0.75,
                pattern_type="continuation",
                direction="bullish"
            )
        elif upper_slope < 0 and abs(lower_slope) < 0.001:
            return Pattern(
                name="Descending Triangle",
                start_idx=min(minima[-3], maxima[-3]),
                end_idx=max(minima[-1], maxima[-1]),
                confidence=0.75,
                pattern_type="continuation",
                direction="bearish"
            )
        elif abs(upper_slope + lower_slope) < 0.001:
            return Pattern(
                name="Symmetrical Triangle",
                start_idx=min(minima[-3], maxima[-3]),
                end_idx=max(minima[-1], maxima[-1]),
                confidence=0.7,
                pattern_type="bilateral",
                direction="neutral"
            )
        return None
    
    def detect_flag_pattern(self) -> Optional[Pattern]:
        """
        Detects flag continuation patterns.
        
        Returns:
            Pattern object if detected, None otherwise
        """
        if len(self.prices) < 20:
            return None
            
        # Check for strong price move (pole)
        pole_start = 0
        pole_end = 10
        pole_move = self.prices[pole_end] - self.prices[pole_start]
        
        # Check if we have a significant move
        if abs(pole_move) / self.prices[pole_start] < 0.05:
            return None
            
        # Check for consolidation (flag)
        flag_prices = self.prices[pole_end:min(pole_end+15, len(self.prices))]
        flag_start = pole_end
        flag_end = flag_start + len(flag_prices) - 1
        
        # Calculate slope of flag
        if len(flag_prices) >= 5:
            flag_slope = np.polyfit(range(len(flag_prices)), flag_prices, 1)[0]
            
            # Bullish flag (pole is up, flag slightly down)
            if pole_move > 0 and flag_slope < 0 and abs(flag_slope) < abs(pole_move) / len(flag_prices) / 2:
                return Pattern(
                    name="Bull Flag",
                    start_idx=pole_start,
                    end_idx=flag_end,
                    confidence=0.75,
                    pattern_type="continuation",
                    direction="bullish"
                )
            # Bearish flag (pole is down, flag slightly up)
            elif pole_move < 0 and flag_slope > 0 and abs(flag_slope) < abs(pole_move) / len(flag_prices) / 2:
                return Pattern(
                    name="Bear Flag",
                    start_idx=pole_start,
                    end_idx=flag_end,
                    confidence=0.75,
                    pattern_type="continuation",
                    direction="bearish"
                )
                
        return None
    
    def detect_cup_and_handle(self) -> Optional[Pattern]:
        """
        Detects cup and handle bullish continuation patterns.
        
        Returns:
            Pattern object if detected, None otherwise
        """
        if len(self.prices) < 30:
            return None
            
        maxima, minima = self.find_local_extrema(window=5)
        
        if len(maxima) < 3 or len(minima) < 1:
            return None
            
        # Check for cup formation
        left_peak = None
        right_peak = None
        cup_bottom = None
        
        for i in range(len(maxima)-1):
            if len(minima) > 0:
                potential_bottom_idx = [m for m in minima if m > maxima[i] and m < maxima[i+1]]
                if potential_bottom_idx:
                    bottom_idx = potential_bottom_idx[0]
                    left_peak = maxima[i]
                    right_peak = maxima[i+1]
                    cup_bottom = bottom_idx
                    break
        
        if left_peak is None or right_peak is None or cup_bottom is None:
            return None
            
        # Check price similarity at cup edges
        left_price = self.prices[left_peak]
        right_price = self.prices[right_peak]
        
        if abs(left_price - right_price) / left_price > self.tolerance:
            return None
            
        # Check for handle (small dip after right peak)
        if right_peak + 5 >= len(self.prices):
            return None
            
        handle_section = self.prices[right_peak:right_peak+10]
        handle_dip = min(handle_section)
        handle_dip_idx = right_peak + np.argmin(handle_section)
        
        if handle_dip < right_price * 0.97 and handle_dip > self.prices[cup_bottom]:
            return Pattern(
                name="Cup and Handle",
                start_idx=left_peak,
                end_idx=handle_dip_idx,
                confidence=0.8,
                pattern_type="continuation",
                direction="bullish"
            )
            
        return None
    
    def detect_market_fractals(self, window=2):
        """
        Detects Bill Williams' fractals in price data.
        
        Args:
            window: Number of candles on each side of the fractal point
            
        Returns:
            Tuple of arrays containing indices of bullish and bearish fractals
        """
        bullish_fractals = []
        bearish_fractals = []
        
        # Detect bullish fractals (bottoms)
        for i in range(window*2, len(self.prices)-window):
            if all(self.prices[i-j] > self.prices[i] for j in range(1, window+1)) and \
               all(self.prices[i+j] > self.prices[i] for j in range(1, window+1)):
                bullish_fractals.append(i)
        
        # Detect bearish fractals (tops)
        for i in range(window*2, len(self.prices)-window):
            if all(self.prices[i-j] < self.prices[i] for j in range(1, window+1)) and \
               all(self.prices[i+j] < self.prices[i] for j in range(1, window+1)):
                bearish_fractals.append(i)
        
        return bullish_fractals, bearish_fractals
    
    def detect_candlestick_patterns(self, ohlc_data, lookback=5):
        """
        Detects Japanese candlestick patterns in OHLC data.
        
        Args:
            ohlc_data: DataFrame with Open, High, Low, Close columns
            lookback: Number of recent candles to analyze
            
        Returns:
            List of detected candlestick patterns
        """
        patterns = []
        
        # Get recent candles
        if len(ohlc_data) < lookback:
            return patterns
            
        recent = ohlc_data[-lookback:].reset_index(drop=True)
        
        # Detect engulfing pattern (last two candles)
        if len(recent) >= 2:
            prev = recent.iloc[-2]
            curr = recent.iloc[-1]
            
            # Bullish engulfing
            if (prev['Close'] < prev['Open'] and  # Previous candle is bearish
                curr['Close'] > curr['Open'] and  # Current candle is bullish
                curr['Open'] < prev['Close'] and  # Current opens below previous close
                curr['Close'] > prev['Open']):    # Current closes above previous open
                
                patterns.append(Pattern(
                    name="Bullish Engulfing",
                    start_idx=len(ohlc_data)-2,
                    end_idx=len(ohlc_data)-1,
                    confidence=0.75,
                    pattern_type="reversal",
                    direction="bullish"
                ))
            
            # Bearish engulfing
            elif (prev['Close'] > prev['Open'] and  # Previous candle is bullish
                  curr['Close'] < curr['Open'] and  # Current candle is bearish
                  curr['Open'] > prev['Close'] and  # Current opens above previous close
                  curr['Close'] < prev['Open']):    # Current closes below previous open
                
                patterns.append(Pattern(
                    name="Bearish Engulfing",
                    start_idx=len(ohlc_data)-2,
                    end_idx=len(ohlc_data)-1,
                    confidence=0.75,
                    pattern_type="reversal",
                    direction="bearish"
                ))
        
        # Detect doji
        if len(recent) >= 1:
            latest = recent.iloc[-1]
            body_size = abs(latest['Close'] - latest['Open'])
            candle_range = latest['High'] - latest['Low']
            
            if candle_range > 0 and body_size / candle_range < 0.1:
                patterns.append(Pattern(
                    name="Doji",
                    start_idx=len(ohlc_data)-1,
                    end_idx=len(ohlc_data)-1,
                    confidence=0.6,
                    pattern_type="indecision",
                    direction="neutral"
                ))
        
        # Detect hammer (single candle)
        if len(recent) >= 1:
            latest = recent.iloc[-1]
            body_size = abs(latest['Close'] - latest['Open'])
            lower_shadow = min(latest['Open'], latest['Close']) - latest['Low']
            upper_shadow = latest['High'] - max(latest['Open'], latest['Close'])
            
            # Hammer criteria
            if (body_size > 0 and 
                lower_shadow >= body_size * 2 and 
                upper_shadow < body_size * 0.5):
                
                pattern_name = "Hammer" if latest['Close'] > latest['Open'] else "Inverted Hammer"
                direction = "bullish" if latest['Close'] > latest['Open'] else "neutral"
                
                patterns.append(Pattern(
                    name=pattern_name,
                    start_idx=len(ohlc_data)-1,
                    end_idx=len(ohlc_data)-1,
                    confidence=0.65,
                    pattern_type="reversal",
                    direction=direction
                ))
        
        return patterns
    
    def validate_pattern_significance(self, pattern, historical_returns, simulations=500):
        """
        Tests if a pattern has predictive power using bootstrap testing.
        
        Args:
            pattern: The pattern to validate
            historical_returns: Array of historical returns
            simulations: Number of bootstrap simulations to run
            
        Returns:
            Updated pattern with statistical validation
        """
        if pattern is None or len(historical_returns) < pattern.end_idx + 10:
            return pattern
            
        # Extract the actual returns after this pattern
        forward_idx = min(pattern.end_idx + 10, len(historical_returns) - 1)
        actual_return = historical_returns[pattern.end_idx:forward_idx].mean()
        
        # Generate bootstrap distribution
        random_returns = []
        for _ in range(simulations):
            # Random sampling from historical data
            valid_range = len(historical_returns) - (forward_idx - pattern.end_idx)
            if valid_range <= 0:
                continue
                
            random_idx = np.random.randint(0, valid_range)
            random_end = min(random_idx + (forward_idx - pattern.end_idx), len(historical_returns))
            random_return = historical_returns[random_idx:random_end].mean()
            random_returns.append(random_return)
        
        # Calculate p-value (significance)
        if len(random_returns) == 0:
            return pattern
            
        if pattern.direction == "bullish":
            p_value = sum(r >= actual_return for r in random_returns) / len(random_returns)
        elif pattern.direction == "bearish":
            p_value = sum(r <= actual_return for r in random_returns) / len(random_returns)
        else:  # neutral
            deviation = abs(actual_return)
            p_value = sum(abs(r) >= deviation for r in random_returns) / len(random_returns)
        
        # Store statistical validation
        if pattern.statistics is None:
            pattern.statistics = {}
            
        pattern.statistics['p_value'] = p_value
        pattern.statistics['expected_return'] = actual_return
        
        # Adjust confidence based on statistical significance
        statistical_confidence = 1.0 - min(p_value, 0.5) * 2  # Scale to 0-1
        pattern.confidence = pattern.confidence * 0.7 + statistical_confidence * 0.3
        
        return pattern
    
    def detect_all_patterns(self, ohlc_data=None):
        """
        Detects all supported patterns in the price data.
        
        Args:
            ohlc_data: Optional OHLC data for candlestick pattern detection
            
        Returns:
            List of all detected patterns
        """
        patterns = []
        
        # Basic patterns
        double_top = self.detect_double_top()
        if double_top: patterns.append(double_top)
        
        double_bottom = self.detect_double_bottom()
        if double_bottom: patterns.append(double_bottom)
        
        head_shoulders = self.detect_head_and_shoulders()
        if head_shoulders: patterns.append(head_shoulders)
        
        inv_head_shoulders = self.detect_inverse_head_and_shoulders()
        if inv_head_shoulders: patterns.append(inv_head_shoulders)
        
        triangle = self.detect_triangle()
        if triangle: patterns.append(triangle)
        
        flag = self.detect_flag_pattern()
        if flag: patterns.append(flag)
        
        cup_handle = self.detect_cup_and_handle()
        if cup_handle: patterns.append(cup_handle)
        
        # Wavelet patterns
        wavelet_patterns = self.wavelet_pattern_detection()
        patterns.extend(wavelet_patterns)
        
        # Candlestick patterns
        if ohlc_data is not None:
            candlestick_patterns = self.detect_candlestick_patterns(ohlc_data)
            patterns.extend(candlestick_patterns)
        
        # Calculate returns for statistical validation
        if len(self.prices) > 1:
            returns = np.diff(self.prices) / self.prices[:-1]
            for i in range(len(patterns)):
                patterns[i] = self.validate_pattern_significance(patterns[i], returns)
        
        return patterns
    
    def optimize_pattern_parameters(self, training_data, target_returns, iterations=30):
        """
        Uses Bayesian optimization to find optimal pattern detection parameters.
        
        Args:
            training_data: Historical price data for training
            target_returns: Future returns for pattern validation
            iterations: Number of optimization iterations
            
        Returns:
            Optimization results
        """
        def objective(params):
            window, tolerance = params
            self.window = int(window)
            self.tolerance = tolerance
            
            detected_patterns = []
            for i in range(0, len(training_data)-100, 20):  # Sample windows
                if i + 100 >= len(training_data):
                    continue
                window_data = training_data[i:i+100]
                window_recognizer = PatternRecognizer(window_data, np.ones_like(window_data))
                window_recognizer.window = self.window
                window_recognizer.tolerance = self.tolerance
                patterns = window_recognizer.detect_all_patterns()
                detected_patterns.extend(patterns)
            
            # Evaluate pattern quality
            pattern_performance = 0
            valid_pattern_count = 0
            
            for pattern in detected_patterns:
                if pattern.end_idx + 10 >= len(target_returns):
                    continue
                    
                future_return = np.mean(target_returns[pattern.end_idx:pattern.end_idx+10])
                
                if (pattern.direction == "bullish" and future_return > 0) or \
                   (pattern.direction == "bearish" and future_return < 0):
                    pattern_performance += abs(future_return) * pattern.confidence
                    valid_pattern_count += 1
                else:
                    pattern_performance -= abs(future_return) * 0.5
            
            if valid_pattern_count == 0:
                return 0
                
            return pattern_performance / max(valid_pattern_count, 1)  # Average performance
        
        # Define the search space
        space = [
            Integer(5, 30, name='window'),
            Real(0.005, 0.05, name='tolerance')
        ]
        
        # Perform optimization
        result = gp_minimize(objective, space, n_calls=iterations, random_state=42)
        
        # Set optimal parameters
        self.window = int(result.x[0])
        self.tolerance = result.x[1]
        
        logger.info(f"Optimized parameters: window={self.window}, tolerance={self.tolerance:.4f}")
        
        return result

class TemplatePatternMatcher:
    """
    Matches price patterns against predefined templates using Dynamic Time Warping.
    """
    def __init__(self):
        # Define template patterns
        self.templates = {
            'double_top': self._create_double_top_template(),
            'double_bottom': self._create_double_bottom_template(),
            'head_shoulders': self._create_head_shoulders_template(),
            'cup_handle': self._create_cup_handle_template(),
            'bullish_flag': self._create_bullish_flag_template()
        }
    
    def _create_double_top_template(self, length=100):
        """Creates an ideal double top pattern template"""
        x = np.linspace(0, 1, length)
        template = np.zeros(length)
        
        # Create the pattern
        peak1_pos, peak2_pos = 0.3, 0.7
        template = 0.4 * np.sin(np.pi * x / 0.5) + 0.5  # Base curve
        
        # Add the peaks
        for i, pos in enumerate(x):
            # First peak
            if abs(pos - peak1_pos) < 0.1:
                template[i] += 0.5 * (1 - 10*abs(pos - peak1_pos))
            
            # Second peak
            if abs(pos - peak2_pos) < 0.1:
                template[i] += 0.5 * (1 - 10*abs(pos - peak2_pos))
        
        # Normalize
        template = (template - template.min()) / (template.max() - template.min())
        return template
    
    def _create_double_bottom_template(self, length=100):
        """Creates an ideal double bottom pattern template"""
        template = 1 - self._create_double_top_template(length)
        return template
    
    def _create_head_shoulders_template(self, length=100):
        """Creates an ideal head and shoulders pattern template"""
        x = np.linspace(0, 1, length)
        template = np.zeros(length)
        
        # Create the pattern
        left_pos, head_pos, right_pos = 0.25, 0.5, 0.75
        
        # Add the shoulders and head
        for i, pos in enumerate(x):
            # Left shoulder
            if abs(pos - left_pos) < 0.1:
                template[i] += 0.3 * (1 - 10*abs(pos - left_pos))
            
            # Head
            if abs(pos - head_pos) < 0.1:
                template[i] += 0.5 * (1 - 10*abs(pos - head_pos))
            
            # Right shoulder
            if abs(pos - right_pos) < 0.1:
                template[i] += 0.3 * (1 - 10*abs(pos - right_pos))
        
        # Normalize
        template = (template - template.min()) / (template.max() - template.min())
        return template
    
    def _create_cup_handle_template(self, length=100):
        """Creates an ideal cup and handle pattern template"""
        x = np.linspace(0, 1, length)
        template = np.zeros(length)
        
        # Create the cup (U shape)
        cup_start, cup_end = 0.1, 0.7
        cup_bottom = (cup_start + cup_end) / 2
        
        for i, pos in enumerate(x):
            if pos >= cup_start and pos <= cup_end:
                # Parabolic cup shape
                cup_val = 1 - 4 * ((pos - cup_bottom) / (cup_end - cup_start))**2
                template[i] = max(0, cup_val)
        
        # Create the handle (slight dip)
        handle_start, handle_end = cup_end, 0.9
        
        for i, pos in enumerate(x):
            if pos >= handle_start and pos <= handle_end:
                # Small rounded dip for handle
                handle_mid = (handle_start + handle_end) / 2
                handle_val = 1 - 0.3 * (1 - ((pos - handle_mid) / ((handle_end - handle_start) / 2))**2)
                template[i] = handle_val
        
        # Set final values
        for i, pos in enumerate(x):
            if pos < cup_start:
                template[i] = 0.8  # Initial level
            elif pos > handle_end:
                template[i] = 1.0  # Final breakout
        
        # Normalize
        template = (template - template.min()) / (template.max() - template.min())
        return template
    
    def _create_bullish_flag_template(self, length=100):
        """Creates an ideal bullish flag pattern template"""
        x = np.linspace(0, 1, length)
        template = np.zeros(length)
        
        # Create the flagpole (sharp rise)
        pole_start, pole_end = 0.1, 0.4
        
        for i, pos in enumerate(x):
            if pos >= pole_start and pos <= pole_end:
                template[i] = (pos - pole_start) / (pole_end - pole_start)
        
        # Create the flag (minor pullback and consolidation)
        flag_start, flag_end = pole_end, 0.8
        
        for i, pos in enumerate(x):
            if pos >= flag_start and pos <= flag_end:
                # Slight downward slope with oscillation
                slope = -0.1
                oscillation = 0.05 * np.sin(20 * np.pi * (pos - flag_start) / (flag_end - flag_start))
                template[i] = template[int(flag_start * length)] + slope * (pos - flag_start) + oscillation
        
        # Breakout
        breakout_start = flag_end
        
        for i, pos in enumerate(x):
            if pos >= breakout_start:
                template[i] = template[int(flag_end * length)] + (pos - breakout_start) / (1 - breakout_start)
        
        # Normalize
        template = (template - template.min()) / (template.max() - template.min())
        return template
    
    def match_pattern(self, price_segment, pattern_type, threshold=0.3):
        """
        Matches a price segment against a pattern template using DTW.
        
        Args:
            price_segment: Array of price values to match
            pattern_type: Type of pattern template to use
            threshold: Distance threshold for pattern detection
            
        Returns:
            Tuple of (is_match, confidence_score)
        """
        if pattern_type not in self.templates:
            return False, 0
            
        template = self.templates[pattern_type]
        
        # Normalize the price segment
        if max(price_segment) > min(price_segment):
            norm_segment = (price_segment - min(price_segment)) / (max(price_segment) - min(price_segment))
        else:
            return False, 0
            
        # Ensure same length as template
        if len(norm_segment) != len(template):
            # Resample to match template length
            indices = np.linspace(0, len(norm_segment)-1, len(template))
            norm_segment = np.interp(indices, np.arange(len(norm_segment)), norm_segment)
        
        # Calculate DTW distance
        distance, _ = fastdtw(norm_segment, template, dist=euclidean)
        normalized_distance = distance / len(template)
        
        # Convert distance to confidence score
        confidence = max(0, 1 - normalized_distance/threshold)
        
        return confidence > 0.5, confidence

class PatternEnsemble:
    """
    Ensemble of different pattern recognition techniques.
    """
    def __init__(self):
        self.detectors = {
            'rule_based': None,  # Will be initialized later with data
            'template_matching': TemplatePatternMatcher(),
            'candlestick': None  # Will be initialized later with data
        }
        self.weights = {
            'rule_based': 0.4,
            'template_matching': 0.3,
            'candlestick': 0.3
        }
    
    def detect_patterns(self, prices, volumes, ohlc_data=None):
        """
        Detects patterns using the ensemble of techniques.
        
        Args:
            prices: Array of price values
            volumes: Array of volume values
            ohlc_data: Optional DataFrame with OHLC data
            
        Returns:
            List of detected patterns
        """
        # Initialize rule-based detector
        self.detectors['rule_based'] = PatternRecognizer(prices, volumes)
        
        all_patterns = []
        
        # Rule-based patterns
        rule_patterns = self.detectors['rule_based'].detect_all_patterns(ohlc_data)
        for pattern in rule_patterns:
            pattern.confidence *= self.weights['rule_based']
            all_patterns.append(pattern)
        
        # Template matching for longer sequences
        matcher = self.detectors['template_matching']
        template_types = [
            ('double_top', 'bearish', 'reversal'),
            ('double_bottom', 'bullish', 'reversal'),
            ('head_shoulders', 'bearish', 'reversal'),
            ('cup_handle', 'bullish', 'continuation'),
            ('bullish_flag', 'bullish', 'continuation')
        ]
        
        for i in range(30, len(prices), 20):  # Slide window through prices
            if i >= len(prices):
                continue
                
            # Extract a window of prices
            start_idx = max(0, i - 100)
            window = prices[start_idx:i]
            
            if len(window) < 50:  # Skip if window too small
                continue
                
            # Check each template
            for template_name, direction, pattern_type in template_types:
                is_match, confidence = matcher.match_pattern(window, template_name)
                
                if is_match:
                    pattern = Pattern(
                        name=f"Template {template_name.replace('_', ' ').title()}",
                        start_idx=start_idx,
                        end_idx=i-1,
                        confidence=confidence * self.weights['template_matching'],
                        pattern_type=pattern_type,
                        direction=direction
                    )
                    all_patterns.append(pattern)
                    
        # Merge overlapping patterns
        merged_patterns = self._merge_overlapping_patterns(all_patterns)
        
        return merged_patterns
    
    def _merge_overlapping_patterns(self, patterns, overlap_threshold=0.7):
        """
        Merges overlapping pattern detections.
        
        Args:
            patterns: List of detected patterns
            overlap_threshold: Minimum overlap ratio to merge patterns
            
        Returns:
            List of merged patterns
        """
        if not patterns:
            return []
            
        # Sort patterns by confidence
        sorted_patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)
        
        merged = []
        used = set()
        
        for i, pattern1 in enumerate(sorted_patterns):
            if i in used:
                continue
                
            curr_pattern = pattern1
            curr_indices = set(range(pattern1.start_idx, pattern1.end_idx + 1))
            
            # Find overlapping patterns
            for j, pattern2 in enumerate(sorted_patterns):
                if i == j or j in used:
                    continue
          
