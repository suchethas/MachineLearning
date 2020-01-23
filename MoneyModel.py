from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
import matplotlib.pyplot as plt
from mesa.datacollection import DataCollector
from random import randrange
from pubsub import pub

class MoneyAgent(Agent):
	# An agent with fixed initial wealth
	def __init__(self, unique_id, model):
		super().__init__(unique_id, model)
		self.wealth = randrange(10)
		self.money_to_transfer = randrange(10)

	def step(self):
		
		if self.wealth < self.money_to_transfer:
			return
		other_agent = self.random.choice(self.model.schedule.agents)
		other_agent.wealth += self.money_to_transfer
		self.wealth -= self.money_to_transfer
		
		print("sender: {}, reciever: {}, USD_Amount: {}".format(self.unique_id, other_agent.unique_id, self.money_to_transfer))
		# ---------- send a message ----------
		pub.sendMessage('Transaction', arg1 = self.money_to_transfer)

		

# ---------- create a listener ----------

def Detector(arg1):
	if arg1 >= 3:
		print("Fraud")

# ---------- register a listener ----------

pub.subscribe(Detector, 'Transaction')

class MoneyModel(Model):
	# A model with some no. of agents
	def __init__(self, N):
		self.num_money_agents = N
		#self.schedule = RandomActivation(self)
		self.schedule = BaseScheduler(self)
		# create agents
		for i in range(self.num_money_agents):
			a = MoneyAgent(i, self)
			# add an agent to schedule
			self.schedule.add(a)

		#self.dc = DataCollector(
					# model_reporters={"Gini": compute_gini},
					#agent_reporters={"Wealth": "wealth"})
	def step(self):
		# Advance the model by one step
		#self.dc.collect(self)
		self.schedule.step()


all_wealth = []
for j in range(1): # run the model this many times
	# Run the model
	model = MoneyModel(10) # create these many money agents
	for i in range(1): # run for so many time steps
		#when schedulers step is called it shuffles the order of the agents
		model.step()
	# store the result
	for a in model.schedule.agents:
		#print(a.unique_id)
		all_wealth.append(a.wealth)

#agent_wealth = model.dc.get_agent_vars_dataframe()
#print(agent_wealth)
# plt.hist(all_wealth)
# plt.show()