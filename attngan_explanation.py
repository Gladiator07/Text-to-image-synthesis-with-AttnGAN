import streamlit as st


def attngan_explanation():
    st.markdown(
        """Text-to-image conversion is a very interesting application of the neural networks (Generative Adverserial Networks to be specific). One of such architecture which used the attention mechanism (borrowed from NLP) was introduced in 2018 by collaboration of Lehigh University, Microsoft Research, Rutgers University, Duke University.  \n\nBefore we dive into the specifics of this research let's first understand what we are dealing with here. The problem is Text-to-Image conversion. 
        So, you give the textual description of a particular object as a description and you will get an image as an output (usually RGB).
""", unsafe_allow_html=True)

    st.markdown("#")
    st.image("img/img-to-text.png", caption="A dog and a kid are playing frisbee")

    st.markdown("#")
    st.markdown(
        """
        When we read the description, we as humans are naturally gifted with this capability of imagination, we are able to visualize this scene without ever actually having to see that precise scene. So that's what we want our models to do: that is, given a description the algorithm should be able to generate an image.

        """
    )