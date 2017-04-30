import train_base_age
import train_base_gender
import train_gender_seeded_by_age
import train_age_seeded_by_gender
import train_gender_seeded_by_age_FREEZED
import train_age_seeded_by_gender_FREEZED
import train_gender_seeded_by_age_DENSE_UNFROZEN
import train_age_seeded_by_gender_DENSE_UNFROZEN



print("\n############################################################")
print("\n\n....SELECT ONE OF THE FOLLOWING OPTIONS....")

print("OPTION 1: Train Base Gender Model")
print("OPTION 2: Train Base Age Model")
print("OPTION 3: Train Gender Model And Train Age Model with initialized weights")
print("OPTION 4: Train Age Model and Train Gender Model with initialized weights")
print("OPTION 5: Train Gender Model And Transfer Learn Age Model")
print("OPTION 6: Train Age Model and Transfer Learn Gender Model")
print("OPTION 7: Train Gender Model and Transfer Learn Gender Model with DENSE layer UNFROZEN")
print("OPTION 8: Train AGE Model and Transfer Learn AGE Model with DENSE layer UNFROZEN")
print("\n############################################################")



print("\nSelected option: ")
option =int(raw_input())

print("\n##########################################################\n")


print("Approx Expected execution time: 40 Mintues")

print("\n##########################################################\n")

#####################
import time
start = time.clock()

######################



if option==1:
    print("\n\nNow Training Base Gender Model")
    train_base_gender.train_model()
elif option==2:
    print("\n\nNow Training Base AGE Model")
    train_base_age.train_model()
elif option==3:
    print("\n\nNow Training Gender Model And Training Age Model with initialized weights")
    train_age_seeded_by_gender.train_model()
elif option==4:
    print("\n\nNow Training Age Model and Training Gender Model with initialized weights")
    train_gender_seeded_by_age.train_model()
elif option==5:
    print("\n\nNow Training Gender Model And Transfer Learn Age Model")
    train_age_seeded_by_gender_FREEZED.train_model()
elif option==6:
    print("\n\nNow Training Age Model and Transfer Learn Gender Model")
    train_gender_seeded_by_age_FREEZED.train_model()
elif option==7:
    print("\n\nNow Training AGE Model and Transfer Learn Gender Model with DENSE layer UNFROZEN")
    train_gender_seeded_by_age_DENSE_UNFROZEN.train_model()
elif option==8:
    print("\n\nNow Training AGE Model and Transfer Learn GENDER Model with DENSE layer UNFROZEN")
    train_age_seeded_by_gender_DENSE_UNFROZEN.train_model()
else: 
    print("\n\nINCORRECT OPTION... TRY AGAIN")


#############################    
total_time = time.clock() - start

print("Total Exceution time: "  + str(total_time) )
#############################
