import shap
import joblib
import matplotlib.pyplot as plt

def explain_model(model, X_sample, feature_names, class_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
   
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_sample,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.savefig('shap_force_plot.png', bbox_inches='tight')
    plt.clf()
    
    
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.savefig('shap_summary.png', bbox_inches='tight')
    
    return shap_values

