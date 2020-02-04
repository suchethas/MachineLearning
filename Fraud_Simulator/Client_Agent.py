from mesa import Agent, Model
from random import randrange
from pubsub import pub
import random

# ---------- RANDOM COUNTRY GENERATOR ----------
def random_country(fname):
	lines = open(fname).read().splitlines()
	return random.choice(lines)

class ClientAgent(Agent):
	def __init__(self, unique_id, model):
		# create a client with id, random units of money, random country
		super().__init__(unique_id, model)
		self.wealth = randrange(100)
		self.country = random_country('countries.txt')

	def step(self):
		# At every step of the model, an agent gives money (if they have it and chosen randomly) 
	 	# to some other agent (other than itself)
		money_to_transfer = randrange(100)
		if self.wealth < money_to_transfer:
			return

		other_agent = self.random.choice(self.model.schedule.agents)
		while other_agent.unique_id == self.unique_id:
			other_agent = self.random.choice(self.model.schedule.agents)

		other_agent.wealth += money_to_transfer
		self.wealth -= money_to_transfer
		
		print("sender: {}, sender country: {}, reciever: {}, reciever country: {}, USD_Amount: {}".format(self.unique_id, self.country, other_agent.unique_id, other_agent.country, money_to_transfer))
		
		# ---------- send a message ----------
		pub.sendMessage('Transaction', w = money_to_transfer, c = other_agent.country) # country here is the receiver country
