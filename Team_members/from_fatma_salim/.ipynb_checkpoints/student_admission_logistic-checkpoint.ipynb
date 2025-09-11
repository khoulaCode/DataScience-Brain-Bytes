{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1071c505-fab8-493c-8b5d-46d97a24ddc2",
   "metadata": {},
   "source": [
    "### Goal: Predict whether a student gets admitted to university based on exam scores.\n",
    "### Algorithm: Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "da097f16-fc44-48c6-b861-213853c5d8ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.6363636363636364\n",
      "\n",
      "Confusion Matrix:\n",
      " [[ 8  3]\n",
      " [ 9 13]]\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "         0.0       0.47      0.73      0.57        11\n",
      "         1.0       0.81      0.59      0.68        22\n",
      "\n",
      "    accuracy                           0.64        33\n",
      "   macro avg       0.64      0.66      0.63        33\n",
      "weighted avg       0.70      0.64      0.65        33\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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SQJAFi9eLImJiZKUlCQiIlu2bBEHBwdZsmSJJCQkSExMjHh7e8u4ceNERCQrK0uqV68uQUFBEhsbK7t375aAgACd2MgwFi1aJPXq1RMRkV9++UW8vb0lOztbREQ2btwo5ubmMmbMGDl16pTExcXJxIkTtct27txZPD09Zd26dZKQkCC//fabrFq1SkRErl+/LmXKlJGRI0fK6dOn5ciRI9KiRQsJDg7WLh8YGCh2dnYyePBgOXPmjPzwww9iY2Mj8+fPFxGR5ORkeemll+SLL76QxMRESUxMFBGRY8eOiZ2dncyYMUPi4+Pljz/+kICAAAkPD9euu02bNlK9enX5888/5dChQ9KkSROxtrbWHsukTExsL4CwsDAxNzcXW1tbsbKyEgACQKZPny4iIp6enrJixQqdZSZMmCCNGzfWvn8yeSxatEiqVq2q/eISEUlPTxdra2vZunWriIi4u7vLqFGj8o0pr2T06quvyqRJk3Tali1bJuXKlRMRka1bt4q5ublcuXJFO33z5s1MbEagSZMmMnPmTBERycjIkDJlysi2bdtERKRx48bSvXv3PJc7e/asANDO+7TRo0dLy5YtddquXLkiAOTs2bMi8jix+fn56RyPn3zyifj5+Wnfe3l55UpGPXr0kPfee0+n7ffffxczMzN5+PChNrb9+/drp58+fVoAMLEpHB80+oIIDg7GvHnzkJqaioULFyI+Ph4ffvghbt68iStXrqBPnz7o27evdv7MzEw4Ojrmua7Dhw/j3LlzsLe312lPS0tDQkICkpKScP36dTRr1qxQMR4+fBgHDx7ExIkTtW1ZWVlIS0tDamoqTp8+jfLly+Oll17STm/cuHGhtkFF7+zZszhw4ADWrVsHALCwsECXLl0QFRWF5s2bIy4uTufYelJcXBzMzc0RGBiY5/TDhw9j586dsLOzyzUtISEBPj4+AIBGjRrpDDE2btwYX3/9NbKysmBubp7vus+dO6czvCgiyM7OxoULFxAfHw8LCwvUq1dPO93X1xelSpV69gdCLzwmtheEra0tKleuDACYNWsWgoODMX78eO11rwULFqBhw4Y6y+T3hZCdnY26devmeb3BxcUFZmbPd+k1Ozsb48eP17n+ksPKykp7zeRJKiU+vvcFs2jRImRmZsLDw0PbJiKwtLTEv//+C2tr63yXfdY04PExERoaiilT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",
      "text/plain": [
       "<Figure size 500x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Load dataset\n",
    "data = pd.read_csv(\"student_admission_dataset.csv\")\n",
    "\n",
    "# Map Admission_Status: Accepted → 1, Rejected → 0\n",
    "data[\"Admission_Status\"] = data[\"Admission_Status\"].map({\"Accepted\": 1, \"Rejected\": 0})\n",
    "\n",
    "# Drop rows with missing Admission_Status\n",
    "data = data.dropna(subset=[\"Admission_Status\"])\n",
    "\n",
    "# Features (X) and Target (y)\n",
    "X = data[[\"GPA\", \"SAT_Score\", \"Extracurricular_Activities\"]]\n",
    "y = data[\"Admission_Status\"]\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Logistic Regression model\n",
    "model = LogisticRegression(max_iter=1000)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Predictions\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nConfusion Matrix:\\n\", confusion_matrix(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n",
    "\n",
    "# Plot Confusion Matrix\n",
    "plt.figure(figsize=(5,4))\n",
    "sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=\"d\", cmap=\"Blues\", \n",
    "            xticklabels=[\"Rejected\", \"Accepted\"], yticklabels=[\"Rejected\", \"Accepted\"])\n",
    "plt.xlabel(\"Predicted\")\n",
    "plt.ylabel(\"Actual\")\n",
    "plt.title(\"Confusion Matrix - Student Admission Prediction\")\n",
    "plt.show()\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "3f9d1037-0601-46ce-bdee-286b73f9f03e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Columns in dataset: ['GPA', 'SAT_Score', 'Extracurricular_Activities', 'Admission_Status']\n",
      "    GPA  SAT_Score  Extracurricular_Activities Admission_Status\n",
      "0  3.46       1223                           8         Rejected\n",
      "1  2.54        974                           8         Rejected\n",
      "2  2.91        909                           9         Rejected\n",
      "3  2.83       1369                           5         Accepted\n",
      "4  3.60       1536                           7         Accepted\n"
     ]
    }
   ],
   "source": [
    "print(\"Columns in dataset:\", data.columns.tolist())\n",
    "print(data.head())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3dd3f297-c4c8-4643-8a7d-5aad7c14a53b",
   "metadata": {},
   "source": [
    "## Interpretation of Results\n",
    "\n",
    "Accuracy = 63.6% → The model predicts correctly about 2 out of 3 cases.\n",
    "\n",
    "Class 0 (Rejected):\n",
    "\n",
    "Precision = 0.47 → When the model says \"Rejected\", it’s correct 47% of the time.\n",
    "\n",
    "Recall = 0.73 → It catches 73% of all truly rejected students.\n",
    "\n",
    "Class 1 (Accepted):\n",
    "\n",
    "Precision = 0.81 → When the model predicts \"Accepted\", it’s correct 81% of the time.\n",
    "\n",
    "Recall = 0.59 → It only finds 59% of the actual accepted students (misses many)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd915493-b110-418a-ae5f-6c6c0099d236",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
