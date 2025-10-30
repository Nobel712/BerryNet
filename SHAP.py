masker = shap.maskers.Image("inpaint_telea", X_test[0].shape)
explainer = shap.Explainer(model, masker, output_names=classes)
explainer
shap_values = explainer(x_test_each_class, outputs=shap.Explanation.argsort.flip[:5])
shap_values.shape
plt.figure(dpi=400)
shap.image_plot(shap_values)
plt.savefig("shap_image_plot.png", dpi=400, bbox_inches='tight')
plt.close()
