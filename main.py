# Import necessary libraries
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

# Full-page layout
st.set_page_config(layout="wide", page_title="Gradient Descent Visualizer")

# Main Title
st.title("")
st.title("Gradient Descent Visualizer")

# CSS for full-page layout and styling (no scrollbars)
st.markdown("""
    <style>
        body {
            font-family: 'serif';  /* Serif font for a mathematical feel */
            background-color: #161748;  /* Dark background */
            color: white;
            width:100%:
            height:100%;
            overflow: hidden;  /* Hide scrollbars */
        }
        .block-container {
            padding: 1rem; /* Padding for page container */
            margin: 0; /* Remove margin */
            max-width: 100%; /* Full page width */
        }
        .stButton>button {
            background-color: #000000;
            color: #ff5e6c;
            border-radius: 8px;
            border: 2px solid #dbb6ee;
        }
        .stTextInput>div>div>input {
            color: white;
            background-color: #161748;
            # border: 2px solid #dbb6ee;
            border-radius: 8px;
        }
        .stNumberInput>div>div>input {
            color: white;
            background-color: #161748;
            border: 2px solid #dbb6ee;
            border-radius: 8px;
        }
        .stPlotlyChart {
            border: 2px solid #dbb6ee;
            border-radius: 15px;
            margin: 0;
            padding: 0;
        }
        .iteration-info {
            color: black;
            font-size: 18px;
            font-weight: bold;
            background-color: #39a0ca;
            padding: 6px;
            border-radius: 8px;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

# Divide the layout into two columns
left_col, right_col = st.columns(2)

# Left column for inputs and buttons
with left_col:
    st.markdown("<div class='component-container'></div>", unsafe_allow_html=True)  # Border for input section
    st.markdown("## Function")
    
    if 'text_input_value' not in st.session_state:
        st.session_state.text_input_value = "x**2 + 3*x + 5"

    # Function buttons
    st.write("Functions you should try (click to auto format):")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("x^2", key="x2"):
            st.session_state.text_input_value = "x**2"
    with col2:
        if st.button("x^3", key="x3"):
            st.session_state.text_input_value = "x**3"
    with col3:
        if st.button("sin(x)", key="sinx"):
            st.session_state.text_input_value = "math.sin(x)"
    with col4:
        if st.button("sin(1/x)", key="sin1x"):
            st.session_state.text_input_value = "math.sin(1/x)"
    with col5:
        if st.button("log(x)", key="logx"):
            st.session_state.text_input_value = "math.log(x)"

    # Custom function input
    st.text_input("## Enter a function of your choice :", value=st.session_state.text_input_value, key="text_input")

    # Starting point input
    start_point = st.number_input("## Start point :", value=2)

    # Learning rate input
    learn_rate = st.number_input("## Learning Rate (η) :", value=0.25)

    # Setup button
    if st.button("Set Up"):
        st.session_state.iteration = 0
        st.session_state.theta_history = [start_point]
        st.session_state.current_fn = st.session_state.text_input_value
        st.write("Setup complete! Click 'Next Iteration' to start.")

# Gradient descent function with error handling
def gradient_descent(fn, start_point, learning_rate, num_iterations):
    theta = start_point
    theta_history = [theta]

    # Define function gradients manually
    def get_gradient(fn, x):
        epsilon = 1e-6
        try:
            if "x**2" in fn:
                return 2 * x  # derivative of x^2
            elif "x**3" in fn:
                return 3 * x**2  # derivative of x^3
            elif "sin(x)" in fn:
                return math.cos(x)  # derivative of sin(x)
            elif "sin(1/x)" in fn:
                return -math.cos(1/x) / (x**2)  # derivative of sin(1/x)
            elif "log(x)" in fn:
                return 1 / x  # derivative of log(x)
            else:
                return 0  # default to 0 if function is unsupported
        except:
            return 0  # Handle undefined behavior

    for _ in range(num_iterations):
        gradient = get_gradient(fn, theta)
        theta = theta - learning_rate * gradient
        if abs(theta) > 1e10:
            theta = np.sign(theta) * 1e10
        theta_history.append(theta)

    return theta_history

def plot(fn, theta_history, iteration):
    # Convert history to float values
    theta_history = [float(theta) for theta in theta_history]
    if not theta_history:
        st.write("No iterations yet. Please click 'Next Iteration'.")
        return

    x = np.linspace(-10, 10, 100)
    y = []

    # Handle edge cases for invalid function evaluations
    for i in x:
        try:
            if "x**2" in fn:
                y.append(i**2)
            elif "x**3" in fn:
                y.append(i**3)
            elif "sin(x)" in fn:
                y.append(math.sin(i))
            elif "sin(1/x)" in fn:
                if i != 0:
                    y.append(math.sin(1/i))
                else:
                    y.append(np.nan)
            elif "log(x)" in fn:
                if i > 0:
                    y.append(math.log(i))
                else:
                    y.append(np.nan)
            else:
                y.append(np.nan)
        except:
            y.append(np.nan)

    # Remove NaN values from x and y
    x_valid = x[~np.isnan(y)]
    y_valid = np.array(y)[~np.isnan(y)]

    last_theta = theta_history[-1]
    meeting_y = None
    try:
        meeting_y = eval(fn.replace('x', str(last_theta))) if 'x' in fn else 0
    except:
        pass

    # Numerical derivative using central difference
    epsilon = 1e-6
    try:
        derivative = (eval(fn.replace('x', str(last_theta + epsilon))) - eval(fn.replace('x', str(last_theta - epsilon)))) / (2 * epsilon)
    except:
        derivative = 0
    slope = derivative
    intercept = meeting_y - slope * last_theta if meeting_y is not None else 0
    tangent_y = slope * x_valid + intercept

    fig = go.Figure(data=[ 
        # Function Line
        go.Scatter(x=x_valid, y=y_valid, mode='lines', name='Function', 
                   line=dict(color='blue')),
        # Gradient Descent Points
        go.Scatter(x=theta_history, 
                   y=[eval(fn.replace('x', str(theta))) for theta in theta_history], 
                   mode='markers', name='Gradient Descent',
                   marker=dict(color='red', size=10)),  # All points are red
        # Tangent Line
        go.Scatter(x=x_valid, y=tangent_y, mode='lines', name='Tangent', 
                   line=dict(color='orange')),
        # Tangent Point (Red)
        go.Scatter(x=[last_theta], y=[meeting_y], mode='markers', name='Tangent Point',
                   marker=dict(color='red', size=12))
    ])

    # Update layout for styling
    fig.update_layout(
        annotations=[
            dict(
                xref='paper', yref='paper', x=0.05, y=0.1,
                xanchor='left', yanchor='bottom',
                text=f"<b>Next Iteration: {iteration}</b>",
                showarrow=False,
                font=dict(size=20, color='black'),
                bgcolor="#f95d9b", borderpad=5, bordercolor="black", borderwidth=2
            ),
            dict(
                xref='paper', yref='paper', x=1, y=0,
                xanchor='right', yanchor='bottom',
                text=f"Current Point: ({last_theta:.6f}, {meeting_y if meeting_y is not None else 'N/A'})",
                showarrow=False,
                font=dict(size=14, color='black'),
                bgcolor="#39a0ca", borderpad=5, bordercolor="black", borderwidth=2
            )
        ],
        xaxis_title='x-axis',
        yaxis_title='y-axis',
        hovermode='x unified',
        xaxis=dict(
            range=[-10, 10], 
            showgrid=True, gridcolor='black', 
            titlefont=dict(color='black'),
            tickfont=dict(color='black')  # Make x-axis numbers black
        ),
        yaxis=dict(
            range=[-10, 10], 
            showgrid=True, gridcolor='black', 
            titlefont=dict(color='black'),
            tickfont=dict(color='black')  # Make y-axis numbers black
        ),
        paper_bgcolor='white',  # White background
        plot_bgcolor='white',   # White plot background
        legend=dict(
            yanchor='top', xanchor='right', x=1, y=0.99, 
            font=dict(color='black')
        ),
        title="Gradient Descent Visualization", titlefont=dict(color='black')
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    return last_theta, meeting_y

def main():
    with right_col:
        if 'iteration' not in st.session_state:
            st.session_state.iteration = 0
            st.session_state.theta_history = [start_point]
            st.session_state.current_fn = st.session_state.text_input_value

        theta_history = st.session_state.theta_history
        iteration = st.session_state.iteration
        current_fn = st.session_state.current_fn

        if st.button("Next Iteration", key="next_iter"):
            iteration += 1
            theta_history = gradient_descent(current_fn, start_point, learn_rate, iteration)
            st.session_state.iteration = iteration
            st.session_state.theta_history = theta_history

        # Plot the function and gradient descent
        last_theta, meeting_y = plot(current_fn, theta_history, iteration)
        
        # Display iteration and point details
        st.markdown(f"## Iteration: {int(iteration)}")
        st.markdown(f"The tangent is meeting the plot at point **({last_theta}, {meeting_y if meeting_y is not None else 'N/A'})**")

# Run the app
if __name__ == "__main__":
    main()

