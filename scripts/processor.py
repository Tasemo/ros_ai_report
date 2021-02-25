#!/usr/bin/env python3

import rospy
import wave
from ros_ai_report.msg import AggregatedData
from ros_ai_report.srv import CommandPrediction, CommandPredictionResponse

def on_audio_data(data):
    if (data.format == "wave"):
        filename = "recording.wav"
        with (wave.open(filename, "wb")) as file:
            file.setnchannels(1)
            file.setsampwidth(data.sample_width)
            file.setframerate(data.sample_rate)
            file.writeframesraw(data.data)
        result = prediction(filename)
        probability = result.probability * 100
        rospy.loginfo("AI node predicted command '%s' with a probability = %i%%", result.command, probability)
    else:
        raise ValueError("The format " + data.format + " is not supported yet!")

if __name__ == '__main__':
    rospy.wait_for_service('command_prediction')
    prediction = rospy.ServiceProxy('command_prediction', CommandPrediction)
    rospy.init_node('processor', anonymous=True)
    rospy.Subscriber('audio_data', AggregatedData, on_audio_data)
    rospy.spin()