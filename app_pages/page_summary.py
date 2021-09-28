

import streamlit as st
import matplotlib.pyplot as plt

def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**Project Terms & Jargons**\n"
        f"* A **customer** is a person who consumes your service or product.\n"
        f"* A **prospect** is a potential customer.\n"
        f"* A **churned** customer is a user who has stopped using your product or service.\n "
        f"* This customer, has a **tenure** level, which is the number of months this person " 
        f"has used our product/service.\n\n"
        f"**Project Dataset**\n"
        f"* The dataset represents a **customer base from a Telco company**, "
        f"containing individual customer data on the products and services "
        f"(like internet type, online security, online backup, tech support), "
        f"account information (like contract type, payment method, monthly charges) "
        f"and profile (like gender, partner, dependents).")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/FernandoRocha88/WalkthroughProject02/blob/main/README.md).")
    

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested to understand the patterns from customer base, "
        f"so the client can learn the most relevant variables that are correlated to a "
        f"churned customer.\n"
        f"* 2 - The client is interested to tell whether or not a given prospect will churn. "
        f"If so, the client is interested to know when. In addition the client is "
        f"interested to know from which cluster this prospect will belong in the customer base, "
        f"and based on that, present potential factors that could mantain and/or bring "
        f"the prospect to a non-churnable cluster."
        )

        

    project_snapshot = plt.imread("pictures/requirements.png")
    st.image(project_snapshot, caption='Representations for Business Requirements 1 and 2, respectively.')

   
