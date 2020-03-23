my_dict = {}
def dicto():
    my_dict = {}
    d_dx =10
    d_dy = 10
    my_dict["d_dx"] = d_dx
    my_dict["d_dy"] = d_dy
    return my_dict
my_dict = dicto()

print(my_dict)
print("d_dx = ",my_dict["d_dx"])
