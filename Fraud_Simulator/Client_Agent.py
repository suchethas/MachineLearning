from mesa import Agent, Model
from random import randrange
from pubsub import pub
import pandas as pd
import random
import os

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

	def custom_data_collector(self):
		pass

	def step(self):
		# At every step of the model, an agent gives money (if they have it and chosen randomly) 
	 	# to some other agent (other than itself)
		client_wealth_before = self.wealth
		money_to_transfer = randrange(100)
		if self.wealth < money_to_transfer:
			return

		other_agent = self.random.choice(self.model.schedule.agents)
		while other_agent.unique_id == self.unique_id:
			other_agent = self.random.choice(self.model.schedule.agents)

		bene_wealth_before = other_agent.wealth
		other_agent.wealth += money_to_transfer
		self.wealth -= money_to_transfer

		store = {self.unique_id : (self.unique_id, self.country, client_wealth_before, other_agent.unique_id, other_agent.country, bene_wealth_before, money_to_transfer, self.wealth, other_agent.wealth)}
		#store_df = DataFrame(store, columns= ['Client Id', 'Client Country', 'Client Wealth Before', 'Bene Id', 'Bene Country', 'Bene Wealth Before', 'Money Transfered', 'Client Wealth After', 'Bene Wealth After'])
		store_df = pd.DataFrame.from_dict(store, orient='index', columns= ['Client Id', 'Client Country', 'Client Wealth Before', 'Bene Id', 'Bene Country', 'Bene Wealth Before', 'Money Transfered', 'Client Wealth After', 'Bene Wealth After'])

		# if file does not exist write header 
		if not os.path.isfile('result.csv'):
		   store_df.to_csv('result.csv', header=['Client Id', 'Client Country', 'Client Wealth Before', 'Bene Id', 'Bene Country', 'Bene Wealth Before', 'Money Transfered', 'Client Wealth After', 'Bene Wealth After'])
		else: # else it exists so append without writing the header
		   store_df.to_csv('result.csv', mode='a', header=False)

		#store_df.to_csv('result.csv', mode='a')
		
		#print("sender: {}, sender country: {}, reciever: {}, reciever country: {}, USD_Amount: {}".format(self.unique_id, self.country, other_agent.unique_id, other_agent.country, money_to_transfer))
		
		# ---------- send a message ----------
		pub.sendMessage('Transaction', w = money_to_transfer, c = other_agent.country) # country here is the receiver country
