import igraph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
'''communities exist, randomly choose state
'''


class LanguageAdoptionModel(object):
	def __init__(self, ALPHA):
		
		
		self.alpha = ALPHA # A's prestige
		print("self alpha is %.2f" %self.alpha)
		self.N = 2500 # network size
		self.Net = nx.Graph() # the network
		self.Net1 = nx.Graph()
		self.Net2 = nx.Graph()
		self.Net3 = nx.Graph()
		# initial agents' states: 1-->A, 2-->B, 3-->AB
		self.State = np.zeros(self.N) # initial agents' states: 1-->A, 2-->B, 3-->AB

		# for plot
		self.position = dict()
		self.fig = plt.figure()

	# initialize networks
	def init_network_others(self, graph_name):
		# graph_name = "communities"
		if(graph_name=="SmallWorld"):
			self.Net = nx.watts_strogatz_graph(self.N, 4, 0.1, seed=seed) # small word network
			self.position=nx.spring_layout(self.Net)
		elif(graph_name=="ScaleFree"):
			self.Net = nx.barabasi_albert_graph(self.N, 4, seed=seed) # complet graph
			self.position=nx.spring_layout(self.Net)	
		elif(graph_name=="completGraph"):
			self.Net = nx.complete_graph(self.N) # complet graph
			self.position=nx.spring_layout(self.Net)
		elif(graph_name=="RandomGraph"):
			self.Net = nx.gnp_random_graph(self.N, p=0.2, seed=seed)
			self.position=nx.spring_layout(self.Net)
		elif(graph_name=="communities"):
			self.Net = nx.Graph()
			Ncommunities = 5
			Sizecommunities = 500
			N_edges = 40 # 2500 nodes: total 1000 noise edges
			assert Ncommunities * Sizecommunities == self.N
			for i in range(Ncommunities):
				self.Net = nx.disjoint_union_all([self.Net, nx.barabasi_albert_graph(Sizecommunities,4, seed=seed)])
			print("#community: %d, edge number: %d" %(Ncommunities ,nx.number_of_edges(self.Net)))
			for i in range(Ncommunities):
				for j in range(N_edges):
					all_comms = set(range(Ncommunities))
					this_comm = set([i])
					another_comm = np.random.choice(list(all_comms - this_comm))
					node1 = np.random.choice(range(i%Ncommunities*Sizecommunities, i%Ncommunities*Sizecommunities+Sizecommunities)) 
					node2 = np.random.choice(range(another_comm%Ncommunities*Sizecommunities, another_comm%Ncommunities*Sizecommunities+Sizecommunities))
					self.Net.add_edge(node1, node2)
			print("#community: %d, edge number: %d" %(Ncommunities ,nx.number_of_edges(self.Net)))
			# g1 = nx.barabasi_albert_graph(self.N_A, 4, seed=seed)			
			# g2 = nx.barabasi_albert_graph(self.N_B, 4, seed=seed)
			# g3 = nx.barabasi_albert_graph(self.N_AB, 4, seed=seed)
			# # g1 = nx.complete_graph(self.N_A)
			# # g2 = nx.complete_graph(self.N_B)
			# # g3 = nx.complete_graph(self.N_AB)
			# # g3 = nx.watts_strogatz_graph(self.N_AB, 4, 0.2, seed=seed)# AB is not fully connected
			# self.Net = nx.disjoint_union_all([g1, g2, g3])
			# self.Net1 = g1
			# self.Net2 = g2
			# self.Net3 = g3
			# N_random_number = 50 # the number of random edges added to the network
			# for i in range(N_random_number):
			# 	e = np.random.choice(self.N, 2, replace=False)
			# 	self.Net.add_edge(e[0], e[1])


			self.position = nx.spring_layout(self.Net)
			# self.initPos()
			# N_12_edges = 10
			# N_13_edges = 20
			# N_23_edges = 20
			# for i in range(N_12_edges):
			# 	node1 = np.random.choice(range(0, self.N_A))
			# 	node2 = np.random.choice(range(self.N_A, self.N_A+self.N_B))
			# 	self.Net.add_edge(node1, node2)
			# for i in range(N_13_edges):
			# 	node1 = np.random.choice(range(0, self.N_A))
			# 	node2 = np.random.choice(range(self.N_A+self.N_B, self.N))
			# 	self.Net.add_edge(node1, node2)
			# for i in range(N_23_edges):
			# 	node1 = np.random.choice(range(self.N_A, self.N_A+self.N_B))
			# 	node2 = np.random.choice(range(self.N_A+self.N_B, self.N))
			# 	self.Net.add_edge(node1, node2)
		else:
			self.init_network(self)

		# State
		print("initializing states...")
		# self.comm_init_State(Ncommunities, Sizecommunities)
		for i in range(0, self.N):
			self.State[i] = np.random.choice([1,2,3])

	def comm_init_State(self, Ncommunities, Sizecommunities):
		for i in range(0, Ncommunities/2):
			for j in range(Sizecommunities):
				rdn = np.random.choice([1,2,3], p = [0.7, 0.25, 0.05])
				self.State[i*Sizecommunities + j] = rdn

		for i in range(Ncommunities/2, Ncommunities):
			for j in range(Sizecommunities):
				rdn = np.random.choice([1,2,3], p = [0.25, 0.7, 0.05])
				self.State[i*Sizecommunities + j] = rdn

	# change with a probability
	def interaction2(self):
		''' TODO: prestige related with density'''
		
		alpha = self.alpha # prestige of language A
		gamma1 = 1.0
		gamma2 = 1.0
		gamma3 = 1.0
		gamma = 1.0 # the volatility coefficient; TODO: gamma could be different in different cases

		for i in range(self.N):
			neighbors = list(self.Net.neighbors(i))
			num_A = 0.0
			num_B = 0.0
			num_all = 0.0
			if(len(neighbors)==0):
				continue
			for nei in neighbors:
				if(self.State[nei]==1):
					num_A += 1.0
				elif(self.State[nei]==2):
					num_B += 1.0
				# else:
				# 	if(self.State[i]==1):
				# 		num_A += 1.0
				# 	elif(self.State[i]==2):
				# 		num_B += 1.0
				# 	else:
				# 		num_A += 1.0
				# 		num_B += 1.0
				num_all += 1.0
			
			# if(self.State[i]==1):
			# 	num_A += 1.0
			# elif(self.State[i]==2):
			# 	num_B += 1.0
			# else:
			# 	num_A += 1.0
			# 	num_B += 1.0
			# num_all += 1.0

			if(self.State[i]==1):
				prob_toAB = (1-alpha)* (num_B/num_all)** gamma1
				randN = np.random.rand()
				if(randN<prob_toAB):
					self.State[i] = 3
			elif(self.State[i]==2):
				prob_toAB = alpha * (num_A/num_all)** gamma2
				randN = np.random.rand()
				if(randN<prob_toAB):
					self.State[i] = 3
			else:
				prob_toA = alpha * (1 - num_B/num_all)** gamma3
				prob_toB = (1-alpha) * (1 - num_A/num_all)** gamma3
				prob_same = 1.0 - (prob_toA + prob_toB)
				self.State[i] = np.random.choice([1,2,3],size = 1, p=[prob_toA,prob_toB,prob_same])[0]


	def updateFig(self, frame):
		ax = self.fig.add_axes([0,0,1,1])
		plt.cla()
		filename = str(frame) + "_image.pdf"
		color_map = {1:'b', 2:'r', 3:'g'}		
		scat = nx.draw(self.Net, pos=self.position, node_color=[color_map[self.State[node] ] for node in self.Net])
		
		self.interaction2()
		return scat

	def initPos(self):
		# self.position=nx.random_layout(self.Net)
		self.position1 = nx.spring_layout(self.Net1)
		self.position2 = nx.spring_layout(self.Net2)
		self.position3 = nx.spring_layout(self.Net3)
		for i in range(0, self.N_A):
			self.position1[i][0] = 0.5 * self.position1[i][0]
			self.position1[i][1] = 0.6 * self.position1[i][1]
		for i in range(0, self.N_B):
			self.position2[i+self.N_A] = np.zeros(2)
			self.position2[i+self.N_A][0] = 0.5 + 0.5 * self.position2[i][0]
			self.position2[i+self.N_A][1] = 0.6 * self.position2[i][1]
			del self.position2[i]
		for i in range(0, self.N_AB):
			self.position3[i+self.N_A+self.N_B] = np.zeros(2)
			self.position3[i+self.N_A+self.N_B][0] = self.position3[i][0]
			self.position3[i+self.N_A+self.N_B][1] = 0.6 + self.position3[i][1]
			del self.position3[i]
		self.position = dict(self.position1.items() + self.position2.items() + self.position3.items())


	def showAnimation(self, Nframe=500):
		ax = self.fig.add_axes([0,0,1,1])
		plt.cla()
		color_map = {1:'b', 2:'r', 3:'g'}
				
		animation = FuncAnimation(self.fig, self.updateFig, interval=50, blit=False, frames=Nframe)
		animation.save("LanguageAdoptionModel.mp4")

	def NumWithTime(self, timeSteps=500):
		num_A = np.zeros(timeSteps+1)
		num_B = np.zeros(timeSteps+1)
		num_AB = np.zeros(timeSteps+1)
		num_A[0] = len(np.where(self.State==1)[0])
		num_B[0] = len(np.where(self.State==2)[0])
		num_AB[0] = len(np.where(self.State==3)[0])
		for t in range(1, timeSteps+1):
			print("time: %d" %t)
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
		# plt.legend(loc='upper right')
		plt.legend()
		axes = plt.gca()
		axes.set_ylim([0,1])
		plt.xlabel('TimeStep', fontsize=16)
		plt.ylabel('Ratio', fontsize=18)
		plt.title("Ratio change in communities (N=2500(5*500), alpha=%.2f, noise edge:200)" %self.alpha)
		# plt.savefig("com_5_0p5_%.2f.pdf" %self.alpha)
		plt.savefig("com_5_0p5_200.pdf")
		# plt.show()
		filename = "ratio_A_%.2f.txt" % (self.alpha)
		np.save(filename, num_A)

if __name__ == "__main__":
	seed = 1024
	np.random.seed(seed)
	# ALPHA = np.arange(0.50)
	ALPHA = [0.5] # language A' prestige
	for i in ALPHA:
		print("alpha is: %.2f" %i)
		model = LanguageAdoptionModel(i)
		print("begin initializing network...")
		model.init_network_others("communities") # which network to use
		# for i in range(50):
		# 	model.updateFig(i)
		
		# visualize the interaction process
		timeSteps = 1000
		action = "figure"# plot 
		if(action=="Animation"):
			print "initializing animation.. "
			model.showAnimation(timeSteps)
		elif(action=="figure"):
			print "number change with time..."
			model.NumWithTime(timeSteps)

	 # the prestige of language A
	

	



	
	
	


