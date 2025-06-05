#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import threading
import sys
import select
import termios
import tty

class KeyboardJoyNode(Node):
    def __init__(self):
        super().__init__('keyboard_joy_node')
        
        # Publisher for Joy messages
        self.joy_publisher = self.create_publisher(Joy, 'joy', 10)
        
        # Timer to publish Joy messages at regular intervals
        self.timer = self.create_timer(0.05, self.publish_joy)  # 20 Hz
        
        # Initialize axes values (8 axes total)
        self.axes = [0.0] * 8
        self.axes[2] = 1.0  # 2nd index (index 1) always 1
        self.axes[5] = 1.0  # 5th index (index 4) always 1
        self.buttons = [0.0] * 8
        
        # Movement increment
        self.increment = 0.05
        
        # Terminal settings for non-blocking input
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        # Start keyboard input thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        self.get_logger().info('Keyboard Joy Node started')
        self.get_logger().info('Controls:')
        self.get_logger().info('  w/s: Axis 0 (left wrist forward/backward)')
        self.get_logger().info('  a/d: Axis 1 (left wrist left/right)')
        self.get_logger().info('  i/k: Axis 3 (right wrist forward/backward)')
        self.get_logger().info('  j/l: Axis 4 (right wrist left/right)')
        self.get_logger().info('  q: Quit')
        
    def keyboard_listener(self):
        """Listen for keyboard input in a separate thread"""
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                try:
                    key = sys.stdin.read(1).lower()
                    self.process_key(key)
                except:
                    break
    
    def process_key(self, key):
        """Process individual key presses"""
        if key == 'q':
            self.get_logger().info('Quitting...')
            rclpy.shutdown()
            return
            
        # Axis 0: w/s keys
        elif key == 'w':
            self.axes[0] = min(1.0, self.axes[0] + self.increment)
        elif key == 's':
            self.axes[0] = max(-1.0, self.axes[0] - self.increment)
            
        # Axis 2: a/d keys (note: this is index 2, not 1, since index 1 is fixed at 1.0)
        elif key == 'a':
            self.axes[1] = min(1.0, self.axes[1] + self.increment)
        elif key == 'd':
            self.axes[1] = max(-1.0, self.axes[1] - self.increment)
            
        # Axis 3: i/k keys
        elif key == 'i':
            self.axes[3] = min(1.0, self.axes[3] + self.increment)
        elif key == 'k':
            self.axes[3] = max(-1.0, self.axes[3] - self.increment)
        
        # Axis 4: j/l keys
        elif key == 'j':
            self.axes[4] = min(1.0, self.axes[4] + self.increment)
        elif key == 'l':
            self.axes[4] = max(-1.0, self.axes[4] - self.increment)
    
    def publish_joy(self):
        """Publish Joy message"""
        joy_msg = Joy()
        joy_msg.header.stamp = self.get_clock().now().to_msg()
        joy_msg.header.frame_id = 'keyboard_joy'
        
        joy_msg.axes = self.axes.copy()
        joy_msg.axes[2] = 1.0  # Always keep 2nd index at 1
        joy_msg.axes[5] = 1.0  # Always keep 5th index at 1
        
        # Empty buttons array
        joy_msg.buttons = []
        
        self.joy_publisher.publish(joy_msg)
    
    def destroy_node(self):
        """Clean up terminal settings when shutting down"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        keyboard_joy_node = KeyboardJoyNode()
        rclpy.spin(keyboard_joy_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, keyboard_joy_node.old_settings)
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()