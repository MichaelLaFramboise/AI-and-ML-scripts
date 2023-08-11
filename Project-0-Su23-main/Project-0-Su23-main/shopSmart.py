# shopSmart.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
Here's the intended output of this script, once you fill it in:

Welcome to shop1 fruit shop
Welcome to shop2 fruit shop
For orders:  [('apples', 1.0), ('oranges', 3.0)] best shop is shop1
For orders:  [('apples', 3.0)] best shop is shop2
"""
from __future__ import print_function
import shop


def shopSmart(orderList, fruitShops):
    """
        orderList: List of (fruit, numPound) tuples
        fruitShops: List of FruitShops
    """
    
    #totalCost1 = 0.0
    #totalCost2 = 0.0
    #totalcosts = 0
    #for shop in fruitShops:
        #cost = sum(numPounds * )
        #totalcosts.append(cost)
        
    #shop1 =  fruitShops[0]
    #for fruit, numPounds in orderList:
    #    totalCost1 += numPounds * dir1[fruit]
        #print(totalCost1)
    #shop2 = fruitShops[1]
    #for fruit, numPounds in orderList:
    #    totalCost2 += numPounds * dir2[fruit]
        #print(totalCost2)
    #totalcostsindex = [totalCost1,totalCost2]
    #minCost = min(totalCost1,totalCost2)
    #bestShopIndex = totalcostsindex.index(minCost)
    #bestShop = fruitShops[bestShopIndex]
     # Initialize list to hold total cost for each shop
    bestShop = None
    minCost = 100000.0
    # Calculate total cost for each shop
    for shop in fruitShops:
        totalCost = 0.0
        for fruit, numPounds in orderList:
            totalCost += numPounds * shop.fruitPrices[fruit]
        if totalCost < minCost:
            minCost = totalCost
            bestShop = shop
        
    # Find the index of the shop with the minimum total cost
    #minCostIndex = totalCosts.index(min(totalCosts))
    # Return the shop with the minimum total cost
    #return fruitShops[minCostIndex]
    return bestShop


if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orders = [('apples', 1.0), ('oranges', 3.0)]
    dir1 = {'apples': 2.0, 'oranges': 1.0}
    shop1 = shop.FruitShop('shop1', dir1)
    dir2 = {'apples': 1.0, 'oranges': 5.0}
    shop2 = shop.FruitShop('shop2', dir2)
    shops = [shop1, shop2]
    print("For orders ", orders, ", the best shop is", shopSmart(orders, shops).getName())
    orders = [('apples', 3.0)]
    print("For orders: ", orders, ", the best shop is", shopSmart(orders, shops).getName())
