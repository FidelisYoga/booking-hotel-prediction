#!/usr/bin/env python
# coding: utf-8

# In[92]:


import streamlit as st
from hotel_inference import HotelBookingInference


# In[93]:


st.set_page_config(page_title="Hotel Booking Cancellation Predictor")
st.title("Hotel Booking Cancellation")
st.markdown("Fill in the booking details below:")


# In[94]:


with st.form("booking_form"):
    no_of_adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
    no_of_children = st.number_input("Children", min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=10, value=1)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, max_value=15, value=2)
    type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.selectbox("Need Parking?", [0, 1])
    room_type_reserved = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=365, value=30)
    arrival_year = st.selectbox("Arrival Year", [2017, 2018])
    arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
    arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
    market_segment_type = st.selectbox("Market Segment", ['Offline', 'Online', 'Corporate', 'Complementary', 'Aviation'])
    repeated_guest = st.selectbox("Repeated Guest?", [0, 1])
    no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=10, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings (Not Canceled)", min_value=0, max_value=10, value=0)
    avg_price_per_room = st.number_input("Avg Price per Room (€)", min_value=0.0, max_value=500.0, value=100.0)
    no_of_special_requests = st.number_input("Special Requests", min_value=0, max_value=5, value=0)

    submitted = st.form_submit_button("Predict")


# In[95]:


if submitted:
    input_data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }

    # Debug print untuk melihat input data
    st.write("Input sent to model:")
    st.write(input_data)
    
    model = HotelBookingInference()
    try:
        prediction = model.predict(input_data)
        st.subheader("Prediction Result:")
        st.success(f"The booking is predicted to be: **{prediction}**")
    except ValueError as e:
        st.error(str(e))

