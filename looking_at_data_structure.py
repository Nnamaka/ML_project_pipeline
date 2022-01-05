
# it is assumed that "housing" is a panda object

# see first few rows in the dataframe
housing.head()

# brief info of the data: no of rows, no of colunms and missing
# or null values
housing.info()

# "ocean_proximity" is a name of a column in the data
# pd.value_counts() gives the mode or number of occurrances of 
# certain value
housing["ocean_proximity"].value_counts()


# pd.describe() calculates and summerizes mean,std,min,max and count
housing.describe()


# below is code and a function for visualizing data: other 
# visualizations are applicable as the method stated here is not
# exhaustive

# for jupyter notebook: no need for this code in new versions
# of jupyter notebook
%matplotlib inline 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick', labelsize=12)

# where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH =  os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join( IMAGES_PATH , fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)



# another quict way to get a feel of the type of data you are
# dealing with is to polot a histogram for each numerical 
# attribute. A histogram show the number of instances( on the
# -vertical axis) that have a given value range( on the horizon
# -tal axis). you can call the hist() method on lthe whole 
# panda dataset

%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
plt.show()

#The hist() method relies on Matplotlib, which in turn relies on a user-specified
#graphical backend to draw on your screen. So before you can plot anything, you need to
#specify which backend Matplotlib should use. The simplest option is to use Jupyter’s
#magic command %matplotlib inline. This tells Jupyter to set up Matplotlib so it uses
#Jupyter’s own backend. Plots are then rendered within the notebook itself. Note that
#calling show() is optional in a Jupyter notebook, as Jupyter will automatically display
#plots when a cell is executed