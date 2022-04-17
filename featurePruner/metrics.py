from sklearn.metrics import mean_squared_error , mean_absolute_error, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance, log_loss, fbeta_score




def rmse(pred, y, w):
	return mean_squared_error(y_true= y, y_pred=pred, sample_weight=w, squared=True)
	

def mae(pred, y, w):
	return mean_absolute_error(y_true= y, y_pred=pred, sample_weight=w)

def poisson_deviance(pred, y, w):
	return mean_poisson_deviance(y_true= y, y_pred=pred, sample_weight=w)


def gamma_deviance(pred, y, w):
	return mean_gamma_deviance(y_true= y, y_pred=pred, sample_weight=w)

def tweedie_deviance(pred, y, w):
	return mean_tweedie_deviance(y_true= y, y_pred=pred, sample_weight=w)

def cross_entropy(pred, y, w):
	return log_loss(y_true= y, y_pred=pred, sample_weight=w)

def fbeta(pred, y, w, beta):
	return fbeta_score(y_true=y, y_pred=pred, sample_weight=w, beta=beta)
