"""
Blink detection module for EyeCursor.
Distinguishes intentional blinks from natural blinking using timing and pattern analysis.
"""

import time
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass
import config


class BlinkType(Enum):
    """Types of blinks detected."""
    NONE = 0
    SINGLE = 1
    DOUBLE = 2


@dataclass
class BlinkEvent:
    """Container for blink event data."""
    blink_type: BlinkType
    timestamp: float
    duration_ms: float


class BlinkDetector:
    """
    Blink detector that distinguishes intentional blinks from natural blinking.

    Detection strategy:
    1. Natural blinks: Brief (100-300ms), involuntary
    2. Intentional blinks: Longer (300ms+) or patterned (double blink)

    We use:
    - Minimum blink duration to filter out natural blinks
    - Cooldown period to prevent multiple triggers
    - Double-blink detection for right-click
    """

    def __init__(self,
                 threshold: float = config.BLINK_THRESHOLD,
                 consecutive_frames: int = config.BLINK_CONSECUTIVE_FRAMES,
                 cooldown_ms: float = config.BLINK_COOLDOWN_MS,
                 double_blink_interval_ms: float = config.DOUBLE_BLINK_INTERVAL_MS):
        """
        Initialize blink detector.

        Args:
            threshold: EAR threshold for blink detection
            consecutive_frames: Frames required to confirm a blink
            cooldown_ms: Minimum time between detected blinks
            double_blink_interval_ms: Maximum time between blinks for double-click
        """
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.cooldown_ms = cooldown_ms
        self.double_blink_interval_ms = double_blink_interval_ms

        # State variables
        self.blink_counter = 0
        self.is_blinking = False
        self.blink_start_time = 0
        self.last_blink_time = 0
        self.last_blink_duration = 0
        self.blink_history = []

        # Callbacks
        self.on_single_blink: Optional[Callable] = None
        self.on_double_blink: Optional[Callable] = None

    def is_eye_closed(self, left_ear: float, right_ear: float) -> bool:
        """
        Check if eye is closed based on EAR values.

        Args:
            left_ear: Left eye aspect ratio
            right_ear: Right eye aspect ratio

        Returns:
            bool: True if eye is considered closed
        """
        # Use average of both eyes (more robust)
        avg_ear = (left_ear + right_ear) / 2
        return avg_ear < self.threshold

    def update(self, left_ear: float, right_ear: float) -> Optional[BlinkEvent]:
        """
        Update blink detection state with new EAR values.

        Args:
            left_ear: Left eye aspect ratio
            right_ear: Right eye aspect ratio

        Returns:
            BlinkEvent if a valid blink was detected, None otherwise
        """
        current_time = time.time()
        closed = self.is_eye_closed(left_ear, right_ear)

        event = None

        if closed:
            if not self.is_blinking:
                # Blink started
                self.is_blinking = True
                self.blink_start_time = current_time
                self.blink_counter = 1
            else:
                # Continue blinking
                self.blink_counter += 1
        else:
            if self.is_blinking:
                # Blink ended - process it
                self.is_blinking = False
                blink_duration = (current_time - self.blink_start_time) * 1000  # Convert to ms

                event = self._process_blink_end(current_time, blink_duration)

        return event

    def _process_blink_end(self, current_time: float, duration_ms: float) -> Optional[BlinkEvent]:
        """
        Process completed blink and determine if it's an intentional action.

        Args:
            current_time: Current timestamp
            duration_ms: Duration of the blink in milliseconds

        Returns:
            BlinkEvent if valid, None otherwise
        """
        # Filter out very short blinks (natural blinks are typically 100-250ms)
        # Intentional blinks should be at least 300ms or part of a double-blink

        time_since_last_blink = (current_time - self.last_blink_time) * 1000

        # Check if this is part of a double blink
        if time_since_last_blink < self.double_blink_interval_ms:
            # Double blink detected
            self.last_blink_time = current_time
            event = BlinkEvent(BlinkType.DOUBLE, current_time, duration_ms)

            if self.on_double_blink:
                self.on_double_blink()

            return event

        # Check cooldown period
        if time_since_last_blink < self.cooldown_ms:
            return None

        # Single blink detected
        self.last_blink_time = current_time
        self.last_blink_duration = duration_ms

        event = BlinkEvent(BlinkType.SINGLE, current_time, duration_ms)

        if self.on_single_blink:
            self.on_single_blink()

        return event

    def get_state_text(self) -> str:
        """Get current blink state for display."""
        if self.is_blinking:
            duration = (time.time() - self.blink_start_time) * 1000
            return f"Blinking: {duration:.0f}ms"
        return ""

    def reset(self):
        """Reset detector state."""
        self.blink_counter = 0
        self.is_blinking = False
        self.blink_start_time = 0
        self.last_blink_time = 0
        self.last_blink_duration = 0


class IntentionalBlinkDetector(BlinkDetector):
    """
    Enhanced detector that uses multiple criteria for intentional blink detection.

    Additional features:
    - Duration-based: Longer blinks = intentional
    - Pattern-based: Double blinks = right click
    - Both eyes closed requirement
    """

    def __init__(self, min_blink_duration_ms: float = 300, **kwargs):
        """
        Initialize intentional blink detector.

        Args:
            min_blink_duration_ms: Minimum duration for intentional blink
            **kwargs: Arguments passed to parent BlinkDetector
        """
        super().__init__(**kwargs)
        self.min_blink_duration_ms = min_blink_duration_ms
        self._pending_single_blink = False
        self._pending_time = 0

    def update(self, left_ear: float, right_ear: float) -> Optional[BlinkEvent]:
        """
        Update with enhanced detection logic.

        We wait to confirm a single blink until we're sure it's not part of a double blink.
        """
        current_time = time.time()
        event = super().update(left_ear, right_ear)

        # Check for pending single blink that wasn't part of double
        if self._pending_single_blink:
            time_since_pending = (current_time - self._pending_time) * 1000
            if time_since_pending > self.double_blink_interval_ms:
                # Single blink confirmed
                self._pending_single_blink = False
                if self.on_single_blink:
                    self.on_single_blink()

        if event:
            if event.blink_type == BlinkType.SINGLE:
                # Mark as pending - might be first of double
                self._pending_single_blink = True
                self._pending_time = current_time
            return event

        return None

    def _process_blink_end(self, current_time: float, duration_ms: float) -> Optional[BlinkEvent]:
        """
        Override to apply minimum duration filter.

        Intentional blinks should be:
        - At least min_blink_duration_ms long, OR
        - Part of a double-blink pattern
        """
        # Natural blinks are quick (100-250ms)
        # Intentional blinks are longer (300ms+)

        time_since_last_blink = (current_time - self.last_blink_time) * 1000

        # Double blink detection (regardless of duration)
        if time_since_last_blink < self.double_blink_interval_ms:
            self._pending_single_blink = False  # Cancel pending single
            self.last_blink_time = current_time
            event = BlinkEvent(BlinkType.DOUBLE, current_time, duration_ms)
            if self.on_double_blink:
                self.on_double_blink()
            return event

        # Check cooldown
        if time_since_last_blink < self.cooldown_ms:
            return None

        # For single blink, check duration
        if duration_ms >= self.min_blink_duration_ms:
            self.last_blink_time = current_time
            self.last_blink_duration = duration_ms
            event = BlinkEvent(BlinkType.SINGLE, current_time, duration_ms)
            # Don't trigger callback yet - wait for double-blick check
            return event

        return None
