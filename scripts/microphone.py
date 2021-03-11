#!/usr/bin/env python3

import rospy
from pynput.keyboard import Key, Listener
from audio_common_msgs.msg import *
from ros_ai_report.msg import AggregatedData

def on_audio(data):
    global recording
    global audio_data
    if (recording):
        audio_data.append(data.data)

def on_audio_info(data):
    global current_audio_info
    current_audio_info = data

if __name__ == "__main__":
    recording = False
    audio_data = []
    current_audio_info = None
    audio_publisher = rospy.Publisher('audio_data', AggregatedData, queue_size=1)
    rospy.init_node("microphone", anonymous=True)
    rospy.Subscriber("audio", AudioData, on_audio)
    rospy.Subscriber("audio_info", AudioInfo, on_audio_info)
    while not rospy.is_shutdown():
        command = input()
        if (command == "r"):
            recording = not recording
            if (not recording):
                audio_msg = AggregatedData()
                audio_msg.data = b''.join(audio_data)
                audio_msg.sample_rate = current_audio_info.sample_rate
                audio_msg.format = current_audio_info.coding_format
                audio_msg.sample_width = int(current_audio_info.sample_format[1:3]) // 8
                audio_publisher.publish(audio_msg)
                audio_data.clear()
