supercategories_content = """\
# Supercategory: Class1, Class2, Class3, ...
Small Animals: mouse, squirrel, rabbit, dog, raccoon
Medium Animals: tiger, bear, lion, panda, zebra
Large Animals: camel, horse, kangaroo, elephant, cow
Aquatic Mammals: whale, shark, dolphin, fish, octopus
Non-Insect Invertebrates: snail, scorpion, lobster, crab, bat
Insects: bee, butterfly, spider, mosquito, bird
Vehicle: bus, bicycle, motorbike, train, pickup_truck
Sky-Vehicle: airplane, flying_saucer, aircraft_carrier, helicopter, hot_air_balloon
Fruits: strawberry, banana, pear, apple, watermelon
Vegetables: carrot, asparagus, mushroom, onion, broccoli
Music: trombone, violin, cello, guitar, clarinet
Furniture: chair, dresser, table, couch, bed
Household Electrical Devices: clock, floor_lamp, telephone, television, keyboard
Tools: saw, axe, hammer, screwdriver, scissors
Clothes & Accessories: bowtie, pants, jacket, sock, shorts
Man-Made Outdoor: skyscraper, windmill, house, castle, bridge
Nature: cloud, bush, ocean, river, mountain
Food: birthday_cake, hamburger, sandwich, pizza, ice_cream
Stationary: calendar, marker, eraser, pencil, map
Household Items: wine_bottle, cup, teapot, frying_pan, wine_glass
"""

# Save the content to a file
file_path = "../data/dn4il/supercategories.txt"
with open(file_path, "w") as file:
    file.write(supercategories_content)
