import pandas as pd 
import csv
import numpy as np

keypoints = 16
subject = [1,6,7,8,9,11]
motion = 'Directions'
## subject target = 5
sources = []
targets = []
for i in subject:
	source_app = []
	target_app = []
	dataframe = pd.read_csv('./'+motion+'/S'+str(i)+'/alldata_joint_source_1.csv',delimiter=None, header=None)
	print(dataframe[0][0][1])
	source = dataframe[0:len(dataframe)-50]
	target = dataframe[50:len(dataframe)]
	target.reset_index(drop=True, inplace=True)
	source_trans = source.transpose()
	target_trans = target.transpose()
	for j in range(len(source)):
		sources.append([x for x in source_trans[j]])
		targets.append([x for x in target_trans[j]])

print(len(sources))
print(len(targets))
print("============================")
print(sources[0])
# print("S1",sources[0])
# print("S1",targets[0])

# print("S6",sources[3035])
# print("S6",targets[3035])

# print("S7",sources[6599])
# print("S7",targets[6599])

#Save to csv file
with open('./'+motion+'/alldata_joint_1_source.csv', mode='w') as csv_file:
	writer = csv.writer(csv_file, delimiter=',')
	for row in range(len(sources)):
		print(" row = ",row)
		writer.writerow(sources[row])

with open('./'+motion+'/alldata_joint_1_target.csv', mode='w') as csv_file:
	writer = csv.writer(csv_file, delimiter=',')
	for row in range(len(targets)):
		print(" row = ",row)
		writer.writerow(targets[row])