import igraph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
'''communities exist, randomly choose state
'''


class LanguageAdoptionModel(object):
	def __init__(self):
		
		
		self.alpha = 0.5 # language A's prestige
		self.Nrow = 50
		self.Ncol = 50
		self.N = 50 * 50 # network size
		self.Net = np.zeros((self.Nrow, self.Ncol)) # the network		
		# initial agents' states: 1-->A, 2-->B, 3-->AB
		self.State = np.zeros((self.Nrow, self.Ncol), dtype=np.int8) # initial agents' states: 1-->A, 2-->B, 3-->AB

		# self.has_position=False
		self.fig = plt.figure()

	# special networks
	def init_network_lattice(self):		
		for i in range(0, self.Nrow):
			for j in range(0, self.Ncol):
				self.State[i,j] = np.random.choice([1,2,3])
		
	

	# change with a probability
	def interaction2(self):
		''' TODO: prestige related with density'''
		
		alpha = self.alpha # prestige of language A
		gamma = 1.0 # the volatility coefficient; TODO: gamma could be different in different cases

		for i in range(self.Nrow):
			for j in range(self.Ncol):
				nei_states = self.countState(i, j)
				num_A = float(nei_states[0])
				num_B = float(nei_states[1])
				num_AB = float(nei_states[2])						
				num_all = num_A + num_B + num_AB	

				if(self.State[i, j]==1):
					prob_toAB = (1-alpha)* (num_B/num_all)** gamma
					randN = np.random.rand()
					if(randN<prob_toAB):
						self.State[i, j] = 3
				elif(self.State[i, j]==2):
					prob_toAB = alpha * (num_A/num_all)** gamma
					randN = np.random.rand()
					if(randN<prob_toAB):
						self.State[i, j] = 3
				else:
					prob_toA = alpha * (1 - num_B/num_all)** gamma
					prob_toB = (1-alpha) * (1 - num_A/num_all)** gamma
					prob_same = 1.0 - (prob_toA + prob_toB)
					self.State[i, j] = np.random.choice([1,2,3],size = 1, p=[prob_toA,prob_toB,prob_same])[0]

	def changeCounter(self, state_count, row_id, col_id):
		if(self.State[row_id, col_id]==1):
			state_count[0] += 1
		elif(self.State[row_id, col_id]==2):
			state_count[1] += 1
		elif(self.State[row_id, col_id]==3):
			state_count[2] += 1
		else:
			print "error in neighbors' state"
			sys.exit(0)

	def countState(self, row_id, col_id):
		state_count = [0, 0, 0] # the number of each state, initialized with 0's
		if(row_id-1>=0):
			if(col_id-1 >= 0):
				self.changeCounter(state_count, row_id-1, col_id-1)
			self.changeCounter(state_count, row_id-1, col_id)
			if(col_id+1 < self.Ncol):
				self.changeCounter(state_count, row_id-1, col_id+1)

		if(row_id+1 < self.Nrow):
			if(col_id-1 >= 0):
				self.changeCounter(state_count, row_id+1, col_id-1)
			self.changeCounter(state_count, row_id+1, col_id)
			if(col_id+1 < self.Ncol):
				self.changeCounter(state_count, row_id+1, col_id+1)

		if(col_id-1 >= 0):
			self.changeCounter(state_count, row_id, col_id-1)
		if(col_id+1 < self.Ncol):
			self.changeCounter(state_count, row_id, col_id+1)

		return state_count

	def updateFig(self, frame):
		nextstate = np.copy(self.State)
		# nextstate = np.ones((self.Nrow, self.Ncol)) * 1
		nextstate = nextstate / 2.0 - 0.5
		# print(nextstate[0,:])
		plt.cla()
		ax = self.fig.gca()
		ax.set_xticks(np.arange(0, self.Nrow, 1))
		ax.set_yticks(np.arange(0, self.Ncol, 1))
		filename = str(frame) + "_image.pdf"
		scat = plt.imshow(np.ones((self.Nrow, self.Ncol))-nextstate, cmap=plt.get_cmap('gray'), interpolation='nearest')
		# if(frame == 0 or frame == 500 or frame == 1000 or frame == 1500 or frame == 2000):
		# 	plt.savefig(filename)
		self.interaction2()
		return scat

	

	def showAnimation(self, Nframe=500):
		# ax = self.fig.add_axes([0.1,0.1, 0.9, 0.9])
		plt.cla()
				
		animation = FuncAnimation(self.fig, self.updateFig, interval=50, blit=False, frames=Nframe)
		animation.save("LanguageAdoptionLattice.mp4")

	def NumWithTime(self, timeSteps=500):
		# alpha = 0.6
		num_A = np.zeros(timeSteps+1)
		num_B = np.zeros(timeSteps+1)
		num_AB = np.zeros(timeSteps+1)
		num_A[0] = len(np.where(self.State==1)[0])
		num_B[0] = len(np.where(self.State==2)[0])
		num_AB[0] = len(np.where(self.State==3)[0])
		for t in range(1, timeSteps+1):
			self.interaction2()
			num_A[t] = len(np.where(model.State==1)[0])
			num_B[t] = len(np.where(model.State==2)[0])
			num_AB[t] = len(np.where(model.State==3)[0])
		num_A = num_A / self.N
		num_B = num_B / self.N
		num_AB = num_AB / self.N
		plt.plot(range(timeSteps+1), num_A, 'b', label='Ratio of Type A')
		plt.plot(range(timeSteps+1), num_B, 'r', label='Ratio of Type B')
		plt.plot(range(timeSteps+1), num_AB, 'g', label='Ratio of Type AB')
		plt.legend(loc='upper right')
		plt.xlabel('TimeStep', fontsize=16)
		plt.ylabel('Ratio', fontsize=18)
		plt.savefig("n_t.pdf")
		# plt.show()


if __name__ == "__main__":
	seed = 1024
	np.random.seed(seed)
	model = LanguageAdoptionModel()
	model.init_network_lattice()
	# for i in range(2002):
	# 	if (i%50 == 0):
	# 		print(i)
	# 	model.updateFig(i)
	
	# visualize the interaction process
	timeSteps = 500
	action = "figure"
	if(action=="Animation"):
		print "initializing animation.. "
		model.showAnimation(timeSteps)
	elif(action=="figure"):
		print "number change with time..."
		model.NumWithTime(timeSteps)	

	



	
	
	


