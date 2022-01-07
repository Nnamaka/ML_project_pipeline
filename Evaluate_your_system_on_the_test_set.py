
# After tweaking your models for a while, you eventually have a 
# system that performs sufficiently well. Now is the time to evaluate
# the final model on the test set. 
#
# Note: After you get the predictors and the labels from your 
# test set, run your "full_pipeline" to transform the data
# call "transform()" not "fit_transform()"- you do not want to
# fit the test set, and evaluate the final model on the test set

final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)

final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# In some cases, such a point estimate of the generalization error will not be
# quite enough to convince you to launch: what if it is just 0.1% better than the
# model currently in production? You might want to have an idea of how precise
# this estimate is. For this, you can compute a 95% confidence interval for the
# generalization error using

from scipy import stats

confidence = 0.95
squared_errors = ( final_predictions - y_test) ** 2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors)))

# If you did a lot of hyperparameter tuning, the performance will usually be
# slightly worse than what you measured using cross-validation (because your
# system ends up fine-tuned to perform well on the validation data and will
# likely not perform as well on unknown datasets). It is not the case in this
# example, but when this happens you must resist the temptation to tweak the
# hyperparameters to make the numbers look good on the test set; the
# improvements would be unlikely to generalize to new data.