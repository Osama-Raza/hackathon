---
title: "Voice Command Processor with Whisper"
sidebar_label: "Voice Command Processor"
description: "Complete example of voice command processing using OpenAI Whisper"
---

# Voice Command Processor with OpenAI Whisper

## Overview

This example demonstrates a complete voice command processing system using OpenAI Whisper for speech recognition. The system captures audio, processes it through Whisper for accurate speech-to-text conversion, and then executes appropriate robot actions based on the recognized commands.

## Prerequisites

Before implementing this system, ensure you have:

- Python 3.8 or higher
- OpenAI Whisper installed (`pip install openai-whisper`)
- Required audio processing libraries (`pip install pyaudio numpy scipy`)
- OpenAI API key (for Whisper API, optional if using local model)

## Complete Implementation

```python
import whisper
import pyaudio
import numpy as np
import wave
import threading
import queue
import time
import os
from typing import Dict, List, Optional, Any
import openai
import json

class WhisperVoiceProcessor:
    def __init__(self, model_size="base", use_api=False, api_key=None):
        """
        Initialize Whisper voice processor

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            use_api: Whether to use OpenAI API instead of local model
            api_key: OpenAI API key (required if use_api=True)
        """
        self.use_api = use_api

        if use_api:
            if not api_key:
                raise ValueError("API key required when using OpenAI API")
            openai.api_key = api_key
            self.model = None
        else:
            print(f"Loading Whisper model: {model_size}")
            self.model = whisper.load_model(model_size)

        # Audio configuration
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16

        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Voice activity detection
        self.silence_threshold = 500  # Adjust based on your environment
        self.silence_duration = 1.0   # Seconds of silence to trigger stop
        self.min_recording_duration = 0.5  # Minimum recording duration

        # Robot action mapping
        self.action_map = {
            'move forward': self._action_move_forward,
            'go forward': self._action_move_forward,
            'move backward': self._action_move_backward,
            'go backward': self._action_move_backward,
            'turn left': self._action_turn_left,
            'turn right': self._action_turn_right,
            'stop': self._action_stop,
            'hello': self._action_greet,
            'hi': self._action_greet,
        }

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for audio stream
        """
        if self.recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def start_listening(self):
        """
        Start listening for voice commands
        """
        print("Starting to listen for voice commands...")

        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )

        self.stream.start_stream()
        self.recording = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        print("Listening... Speak now!")

    def stop_listening(self):
        """
        Stop listening and clean up
        """
        print("Stopping voice processor...")
        self.recording = False

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)

        self.audio.terminate()
        print("Voice processor stopped.")

    def _process_audio(self):
        """
        Process audio chunks and detect voice activity
        """
        while self.recording:
            try:
                # Collect audio data
                audio_data = self._collect_audio()

                if audio_data:
                    # Process with Whisper
                    recognized_text = self._recognize_speech(audio_data)

                    if recognized_text:
                        print(f"Recognized: {recognized_text}")

                        # Process the command
                        self._process_command(recognized_text)

            except Exception as e:
                print(f"Error in audio processing: {e}")
                time.sleep(0.1)

    def _collect_audio(self) -> Optional[bytes]:
        """
        Collect audio until silence is detected
        """
        frames = []
        silence_start_time = None
        recording_start_time = time.time()

        while self.recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                frames.append(chunk)

                # Convert to numpy array for analysis
                audio_array = np.frombuffer(chunk, dtype=np.int16)
                amplitude = np.abs(audio_array).mean()

                if amplitude < self.silence_threshold:
                    # Potential silence detected
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time >= self.silence_duration:
                        # Sufficient silence detected
                        recording_duration = time.time() - recording_start_time

                        if recording_duration >= self.min_recording_duration:
                            # Valid recording, return it
                            return b''.join(frames)
                        else:
                            # Too short, continue recording
                            silence_start_time = None
                else:
                    # Audio detected, reset silence timer
                    silence_start_time = None

            except queue.Empty:
                continue

        return None if not frames else b''.join(frames)

    def _recognize_speech(self, audio_data: bytes) -> Optional[str]:
        """
        Recognize speech using Whisper
        """
        try:
            if self.use_api:
                # Save audio to temporary file for API
                temp_filename = "temp_recording.wav"
                self._save_audio_to_wav(audio_data, temp_filename)

                with open(temp_filename, "rb") as audio_file:
                    transcript = openai.Audio.transcribe("whisper-1", audio_file)

                # Clean up temp file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

                return transcript.get("text", "").strip()
            else:
                # Process with local Whisper model
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize

                # Transcribe
                result = self.model.transcribe(audio_array)
                return result.get("text", "").strip()

        except Exception as e:
            print(f"Speech recognition error: {e}")
            return None

    def _save_audio_to_wav(self, audio_data: bytes, filename: str):
        """
        Save audio data to WAV file
        """
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)

    def _process_command(self, text: str):
        """
        Process recognized command and execute appropriate action
        """
        text_lower = text.lower().strip()

        # Find matching action
        for command, action_func in self.action_map.items():
            if command in text_lower:
                print(f"Executing command: {command}")
                result = action_func(text_lower)

                # Speak response if available
                if result:
                    self._speak_response(result)

                return

        # No matching command found
        print(f"Command not recognized: {text}")
        self._speak_response("I didn't understand that command.")

    def _action_move_forward(self, command: str) -> str:
        """
        Handle move forward command
        """
        # Extract distance if specified
        import re
        distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(meters?|m)?', command)
        distance = float(distance_match.group(1)) if distance_match else 1.0

        print(f"Moving forward {distance} meters")
        # In a real robot, this would send movement commands
        # self.robot.move_forward(distance)

        return f"Moving forward {distance} meters."

    def _action_move_backward(self, command: str) -> str:
        """
        Handle move backward command
        """
        import re
        distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(meters?|m)?', command)
        distance = float(distance_match.group(1)) if distance_match else 1.0

        print(f"Moving backward {distance} meters")
        # self.robot.move_backward(distance)

        return f"Moving backward {distance} meters."

    def _action_turn_left(self, command: str) -> str:
        """
        Handle turn left command
        """
        import re
        angle_match = re.search(r'(\d+(?:\.\d+)?)\s*(degrees?|deg)?', command)
        angle = float(angle_match.group(1)) if angle_match else 90.0

        print(f"Turning left {angle} degrees")
        # self.robot.turn_left(angle)

        return f"Turning left {angle} degrees."

    def _action_turn_right(self, command: str) -> str:
        """
        Handle turn right command
        """
        import re
        angle_match = re.search(r'(\d+(?:\.\d+)?)\s*(degrees?|deg)?', command)
        angle = float(angle_match.group(1)) if angle_match else 90.0

        print(f"Turning right {angle} degrees")
        # self.robot.turn_right(angle)

        return f"Turning right {angle} degrees."

    def _action_stop(self, command: str) -> str:
        """
        Handle stop command
        """
        print("Stopping robot")
        # self.robot.stop()

        return "Robot stopped."

    def _action_greet(self, command: str) -> str:
        """
        Handle greeting command
        """
        import random
        responses = [
            "Hello! How can I help you?",
            "Hi there! Ready to assist.",
            "Greetings! What would you like me to do?"
        ]
        response = random.choice(responses)

        print(f"Greeting: {response}")

        return response

    def _speak_response(self, text: str):
        """
        Speak response using text-to-speech
        """
        # This is a placeholder - in a real system you'd use a TTS engine
        print(f"Speaking: {text}")
        # Example with pyttsx3:
        # import pyttsx3
        # engine = pyttsx3.init()
        # engine.say(text)
        # engine.runAndWait()

# Example usage with ROS 2 integration
class WhisperROS2Bridge:
    def __init__(self):
        """
        Bridge between Whisper voice processor and ROS 2
        """
        # Initialize Whisper processor
        self.whisper_processor = WhisperVoiceProcessor(model_size="base")

        # ROS 2 initialization would go here
        # self.node = rclpy.create_node('whisper_voice_bridge')
        # self.cmd_vel_publisher = self.node.create_publisher(Twist, 'cmd_vel', 10)

        print("Whisper-ROS2 Bridge initialized")

    def start_bridge(self):
        """
        Start the voice processing bridge
        """
        try:
            self.whisper_processor.start_listening()

            # Keep running
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("Stopping bridge...")
            self.whisper_processor.stop_listening()

def main():
    """
    Main function to run the voice command processor
    """
    print("Whisper Voice Command Processor")
    print("=" * 40)

    # Initialize processor
    # For local model (recommended for privacy):
    processor = WhisperVoiceProcessor(model_size="base")

    # For OpenAI API (requires API key):
    # processor = WhisperVoiceProcessor(use_api=True, api_key="your-api-key")

    try:
        # Start listening
        processor.start_listening()

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
        processor.stop_listening()
        print("Goodbye!")

if __name__ == "__main__":
    main()
```

## Advanced Features

### Wake Word Detection

```python
import vosk
import json

class WakeWordDetector:
    def __init__(self, wake_words=["robot", "hey robot", "robbie"]):
        """
        Initialize wake word detector using Vosk
        """
        self.wake_words = [word.lower() for word in wake_words]
        self.listening_for_wake = True

        # You would initialize Vosk model here
        # self.model = vosk.Model("path/to/vosk-model")
        # self.recognizer = vosk.KaldiRecognizer(self.model, 16000)

    def detect_wake_word(self, audio_data: bytes) -> bool:
        """
        Detect if wake word is present in audio
        """
        # Process with Vosk
        # if self.recognizer.AcceptWaveform(audio_data):
        #     result = self.recognizer.Result()
        #     text = json.loads(result).get("text", "").lower()
        #
        #     for wake_word in self.wake_words:
        #         if wake_word in text:
        #             return True

        # Simplified version for demonstration
        return True  # Always return True for demo purposes
```

### Context-Aware Processing

```python
class ContextAwareProcessor(WhisperVoiceProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Context tracking
        self.context = {
            'current_task': None,
            'user_preferences': {},
            'robot_state': {
                'location': {'x': 0, 'y': 0},
                'battery': 100,
                'orientation': 0
            }
        }

        # Task-specific command maps
        self.task_command_maps = {
            'navigation': {
                'go to kitchen': self._navigate_to_kitchen,
                'return to base': self._return_to_base,
            },
            'manipulation': {
                'pick up object': self._pick_up_object,
                'place object': self._place_object,
            }
        }

    def _process_command_with_context(self, text: str):
        """
        Process command considering current context
        """
        text_lower = text.lower().strip()

        # Check if we're in a specific task context
        current_task = self.context['current_task']

        if current_task and current_task in self.task_command_maps:
            # Check task-specific commands first
            task_commands = self.task_command_maps[current_task]
            for command, action_func in task_commands.items():
                if command in text_lower:
                    return action_func(text_lower)

        # Fall back to general commands
        for command, action_func in self.action_map.items():
            if command in text_lower:
                return action_func(text_lower)

        return "Command not recognized in current context."
```

### Error Recovery and Fallback

```python
class RobustWhisperProcessor(WhisperVoiceProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Fallback mechanisms
        self.fallback_methods = [
            self._try_smaller_model,
            self._try_api_fallback,
            self._try_offline_recognition
        ]

        # Error tracking
        self.error_count = 0
        self.error_threshold = 5

    def _recognize_speech_with_fallback(self, audio_data: bytes) -> Optional[str]:
        """
        Recognize speech with multiple fallback options
        """
        try:
            # Primary recognition
            result = self._recognize_speech(audio_data)
            if result:
                self.error_count = 0  # Reset error count on success
                return result
        except Exception as e:
            print(f"Primary recognition failed: {e}")
            self.error_count += 1

        # Try fallback methods if error threshold exceeded
        if self.error_count >= self.error_threshold:
            for fallback_method in self.fallback_methods:
                try:
                    result = fallback_method(audio_data)
                    if result:
                        self.error_count = 0  # Reset on success
                        return result
                except Exception as e:
                    print(f"Fallback method failed: {e}")
                    continue

        return None

    def _try_smaller_model(self, audio_data: bytes) -> Optional[str]:
        """
        Try with a smaller Whisper model
        """
        if self.model_size != 'tiny':
            # Temporarily use smaller model
            original_model = self.model
            self.model = whisper.load_model('tiny')

            try:
                result = self._recognize_speech(audio_data)
                return result
            finally:
                self.model = original_model

        return None

    def _try_api_fallback(self, audio_data: bytes) -> Optional[str]:
        """
        Try using OpenAI API as fallback
        """
        if hasattr(self, 'api_key'):
            temp_use_api = self.use_api
            self.use_api = True

            try:
                result = self._recognize_speech(audio_data)
                return result
            except Exception:
                pass
            finally:
                self.use_api = temp_use_api

        return None
```

## Testing the Implementation

```python
import unittest
from unittest.mock import Mock, patch

class TestWhisperVoiceProcessor(unittest.TestCase):
    def setUp(self):
        # Create processor with mock components for testing
        self.processor = WhisperVoiceProcessor(model_size="tiny")

    @patch('whisper.load_model')
    def test_recognize_speech(self, mock_load_model):
        # Mock the Whisper model
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "move forward"}
        mock_load_model.return_value = mock_model

        # Test with mock audio data
        mock_audio_data = b"fake_audio_data"

        result = self.processor._recognize_speech(mock_audio_data)
        self.assertEqual(result, "move forward")

    def test_action_mapping(self):
        # Test that commands map to correct actions
        processor = self.processor

        # Test move forward
        result = processor._action_move_forward("move forward 2 meters")
        self.assertIn("2 meters", result)

        # Test turn left
        result = processor._action_turn_left("turn left 45 degrees")
        self.assertIn("45 degrees", result)

    def test_command_processing(self):
        # Test command processing with context
        processor = self.processor

        # Mock the speak response method
        processor._speak_response = Mock()

        # Process a command
        processor._process_command("move forward")

        # Verify the action was called
        processor._speak_response.assert_called()

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
```

## Setup and Installation

Create a requirements file for the voice command processor:

```txt
# requirements.txt
openai-whisper
pyaudio
numpy
scipy
torch>=1.10.0
openai
vosk  # For wake word detection
pyttsx3  # For text-to-speech
rclpy  # For ROS 2 integration (if using ROS 2)
```

## Performance Considerations

For production use, consider these performance optimizations:

1. **Model Selection**: Use appropriate model size for your hardware
2. **Batch Processing**: Process multiple audio chunks together when possible
3. **Caching**: Cache results for frequently used commands
4. **Asynchronous Processing**: Use async/await for non-blocking operations
5. **Memory Management**: Clear audio buffers regularly to prevent memory leaks

## Security and Privacy

When implementing voice processing systems:

1. **Data Encryption**: Encrypt audio data in transit and at rest
2. **API Key Security**: Never hardcode API keys; use environment variables
3. **Local Processing**: Prefer local models over cloud APIs for sensitive applications
4. **User Consent**: Obtain explicit consent for voice data processing
5. **Data Retention**: Implement policies for deleting voice data after processing

## Conclusion

The Whisper-based voice command processor provides a robust foundation for natural human-robot interaction. With proper implementation, it enables intuitive voice control of robotic systems while maintaining privacy and performance considerations. The modular design allows for easy integration with various robotic platforms and can be extended with additional features like wake word detection, context awareness, and error recovery.