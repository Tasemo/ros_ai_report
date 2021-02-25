#!/usr/bin/env python3

import os
import torch
import torchaudio
import pickle
import rospy
from ros_ai_report.model import Model
from ros_ai_report.srv import CommandPrediction, CommandPredictionResponse

def predictCommand(req):
    waveform, sample_rate = torchaudio.load(req.filename)
    if (sample_rate != metadata["sample_rate"]):
        waveform = torchaudio.transforms.Resample(sample_rate, metadata["sample_rate"])(waveform)
    result = model(waveform.unsqueeze(0))
    probability = torch.exp(result.max()).item()
    command = metadata["labels"][result.argmax().squeeze()]
    return CommandPredictionResponse(command, probability)

if __name__ == "__main__":
    current_folder = os.path.dirname(__file__)
    with open(current_folder + "/../model/metadata", "rb") as file:
        metadata = pickle.load(file)
    state = torch.load(current_folder + "/../model/trainedModel.pt")
    model = Model(len(metadata["labels"]))
    model.load_state_dict(state)
    model.eval()
    rospy.init_node('ai')
    rospy.Service('command_prediction', CommandPrediction, predictCommand)
    rospy.spin()
