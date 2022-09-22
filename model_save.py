import torch


## save model for quickly loading
model_path = '/lengyu.yb/logs/mmsports2022/exp07_swa/swa_model_148.pth'
checkpoint = torch.load(model_path, map_location='cpu')
print(checkpoint.keys())
weights = checkpoint['state_dict']
state_dict = {"state_dict": weights}
torch.save(state_dict, '/lengyu.yb/logs/mmsports2022/exp07_swa/swa_model_148_mms.pth')
