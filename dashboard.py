import streamlit as st
import time, json
import numpy as np
import altair as alt
import pandas as pd
import Robogame as rg
import networkx as nx
import matplotlib.pyplot as plt
import collections

# Procedure
# If running locally, uncomment line 34 and comment out line 35-37

# On one terminal window
# $cd /Users/prajnajiang/Documents/si649/si649robogames-main/server
# $python api.py -d ./example1 -s -t1s bob examplematch1
# On another terminal window
# $cd /Users/prajnajiang/Documents/si649/si649robogames-main/clients
# $streamlit run streamlit_test_wdingyu_final.py
# This will launch our dashboard

# Set the page to wide for better view
st.set_page_config(layout="wide")

# Create the game and mark it as ready
@st.cache(allow_output_mutation=True)
def prepGame():
    initial_status.write("prepping game...")
    print('prepping game')
    game = rg.Robogame("bob")
    # game = rg.Robogame("stats", 
    #     server='roboviz.games', port=5000,
    #     multiplayer=True)
    game.setReady()
    return(game)


@st.cache(ttl=3)
def getHintData():
    toReturn = pd.DataFrame(game.getAllPredictionHints())
    initial_status.write("Getting hints, we now have "+str(len(toReturn))+" hints")
    return(toReturn)


@st.cache(ttl=3)
def getAllPartHintsData():
    toReturn = pd.DataFrame(game.getAllPartHints())
    initial_status.write("Getting part hints, we now have "+str(len(toReturn))+" hints")
    return(toReturn)


if 'robot_data' not in st.session_state.keys():
    robot_data = list(np.arange(1,101))
    st.session_state['robot_data'] = robot_data
else:
    robot_data = st.session_state['robot_data']

st.title("Team 2135 Robogames dashboard")
title0 = st.empty()

def robot_checkbox_container(dt):
    '''
    Create the checkbox for each robot id
    '''
    with st.expander("Robots to track:"):
        for i in np.arange(0,10):
            cols = st.columns(10)
            for j in np.arange(1,11):
                foo = i*10+j
                with cols[j-1]:
                    st.checkbox(str(foo), key='dynamic_robot_checkbox_' + str(foo))
  

def get_selected_robot_checkboxes():
    '''
    Get the selected robot id 
    '''
    return [i.replace('dynamic_robot_checkbox_','') for i in st.session_state.keys() if i.startswith('dynamic_robot_checkbox_') and st.session_state[i]]

title0.write('#### Input request to hacker')
robot_checkbox_container(robot_data)
st.write("tell hacker to track robots as below:",list(get_selected_robot_checkboxes()))

if 'feature_data' not in st.session_state.keys():
    feature_data = ["Astrogation Buffer Length", "InfoCore Size", "AutoTerrain Tread Count", 
                    "Polarity Sinks", "Cranial Uplink Bandwidth", "Repulsorlift Motor HP", 
                    "Sonoreceptors", "Arakyd Vocabulator Model", "Axial Piston Model", 
                    "Nanochip Model"]
    st.session_state['feature_data'] = feature_data
else:
    feature_data = st.session_state['feature_data']


def feature_checkbox_container(dt):
    '''
    Create the check box for feature selecting 
    '''
    with st.expander("Features to track:"):
        for feature in feature_data:
            st.checkbox(feature, key='dynamic_feature_checkbox_' + feature)
 

def get_selected_feature_checkboxes():
    '''
    Get the selected feature
    '''
    return [i.replace('dynamic_feature_checkbox_','') for i in st.session_state.keys() if i.startswith('dynamic_feature_checkbox_') and st.session_state[i]]

feature_checkbox_container(feature_data)
st.write("tell hacker to track features as below:",list(get_selected_feature_checkboxes()))

initial_status = st.empty()
status = st.empty()
currentTime = st.empty()
currentTime1 = st.empty()
title1 = st.empty()
predVis = st.empty()
partVis = st.empty()
predVis1 = st.empty()
title2 = st.empty()
scatVis = st.empty()

# Wait for both players to be ready
game = prepGame()
#game = rg.Robogame("bob")
#game.setReady()

while(True):
    gametime = game.getGameTime()
    timetogo = gametime['gamestarttime_secs'] - gametime['servertime_secs']

    if ('Error' in gametime):
        initial_status.write("Error"+str(gametime))
        break
    if (timetogo <= 0):
        initial_status.write("Let's go!")
        break
    initial_status.write("waiting to launch... game will start in " + str(int(timetogo)))
    time.sleep(1) # sleep 1 second at a time, wait for the game to start

all_part_hints_data = st.empty()
robotInterests = []
featureInterests = []

while(True):
    currentRobotInterests = list(get_selected_robot_checkboxes())
    if collections.Counter(currentRobotInterests) != collections.Counter(robotInterests):
        # only update if things have changed
        game.setRobotInterest(currentRobotInterests)
        robotInterests = currentRobotInterests
        print(list(get_selected_robot_checkboxes()))

    currentFeatureInterests = list(get_selected_feature_checkboxes())
    if collections.Counter(currentFeatureInterests) != collections.Counter(featureInterests):
        # only update if things have changed
        game.setPartInterest(currentFeatureInterests)
        featureInterests = currentFeatureInterests
        print(list(get_selected_feature_checkboxes()))
  
    #getAllPartHintsData()
    
    ######----------Social NetWork First Part Start-------------------#####
    ## Construct the social network info dataset
    # grab the networks
    network = game.getNetwork()

    # calculate the popularity using social netwrok: sum the connection
        
    node = {}
    for item in network["links"]:
        if item["source"] not in node:
            node[item["source"]] = 1
        else:
            node[item["source"]] += 1
    for item in network["links"]: 
        if item["target"] not in node:
            node[item["target"]] = 1
        else:
            node[item["target"]] +=1
    node = dict(sorted(node.items()))

    # social network linkage dataset
    source=[]
    target=[]
    for item in network["links"]:
        if item["source"] not in source:
            source.append(item["source"])
        if item["target"] not in target:
            target.append(item["target"])
    relation = pd.DataFrame(network["links"])

    #Create the network graph using networkx
    node_list = set(source+target)
    G = nx.Graph() #Use the Graph API to create an empty network graph object
        
    #Add nodes and edges to the graph object
    for i in node_list:
        G.add_node(i)
    for i,j in relation.iterrows():
        G.add_edges_from([(j["source"],j["target"])])  

    #####----------Design the scatter plot for number guess and present the robot producitivity-------------------####
    #find the unique expiration time
    robots = game.getRobotInfo()
    uniqueExpireTime = pd.unique(robots[robots['id']<100]['expires']).tolist()
    uniqueExpireTime.append(0)
    uniqueExpireTime.sort()

    def productivity_graph(robots, nodes):
        """
        Create a productivity bar plot. Return the viz.
        For each idx, get the productivity of everything in the "nodes" list.
        """
        base = alt.Chart(robots).mark_bar().transform_filter(
            # Exclude the ones w NaN productivity
            # Ref: https://stackoverflow.com/questions/72303847/python-altair-mark-line-ignore-nan-instead-of-skipping-them-or-treatin
            "isValid(datum.Productivity)"
        ).transform_filter(
            alt.FieldOneOfPredicate(field ='id', oneOf = nodes)
        ).transform_window(
            sort=[alt.SortField("Productivity", order="descending")],
            productivity_rank="rank(*)"
        ).encode(
            x=alt.X("Productivity:Q", scale=alt.Scale(domain=[-100, 100])),
            y=alt.Y("id:O",
                    sort=alt.EncodingSortField(
                        field="productivity_rank",
                        order="ascending"
                    ))
        )

        bars = base.mark_bar(color="blue", height=7)

        # If positive productivity,
        # layer on top of the orignal plot (so have diff bar col)
        bars_pos = base.mark_bar(
            color="orange", height=7
        ).transform_filter(
            alt.datum.Productivity > 0
        )

        text_pos = bars.mark_text(
            align="left",
            baseline="middle",
            color="darkblue",
            dx=7
        ).transform_filter(
            alt.datum.Productivity > 0
        ).encode(
            # Round to the nearest integer
            text=alt.Text("Productivity", format=',.0f')
        )

        text_neg = bars.mark_text(
            align="left",
            baseline="middle",
            color="darkblue",
            dx=-20
        ).transform_filter(
            alt.datum.Productivity <= 0
        ).encode(
            # Round to the nearest integer
            text=alt.Text("Productivity", format=',.0f')
        )

        prod_viz = (bars + bars_pos +
            text_pos + text_neg).properties(width=200)
        return prod_viz

    tree = game.getTree()
    genealogy = nx.tree_graph(tree)
    
    # renew when there is a new robot declared

    # run 100 times
    robotInfo = pd.DataFrame(game.getRobotInfo())
    uniqueExpireTime = pd.unique(robotInfo[robotInfo['id']<100]['expires'])
    uniqueExpireTime.sort()

    for k in np.arange(0,101):
        # sleep 5 seconds
        for t in np.arange(0,5):
            status.write("Seconds to next hack: " + str(5-t))
            time.sleep(1)

        # update the hints
        game.getHints()

        # create a dataframe for the time prediction hints
        predictionHints = pd.DataFrame(game.getAllPredictionHints())

        # Get the current game time (this will not change even if the app reruns)
        currentTimeGame = game.getGameTime()['curtime']
        currentTime.write('Current planet time: ' + str(currentTimeGame))
        uniqueExpireTimeGame = uniqueExpireTime[uniqueExpireTime > currentTimeGame+1]

        # Use count to control # of plots on each row
        count = 0

        for times in uniqueExpireTimeGame:
            node_idx = robotInfo[robotInfo['expires'] == times]['id'].values

            for idx in node_idx:
                # Store child, sibling, and parent id's
                child_id = []
                sibling_id = []
                parent_id = []

                # Children
                for i in list(nx.dfs_edges(genealogy,idx)):
                    if(i[0] == idx):
                        child_id.append(i[1])

                # Parent
                parent_id.append(list(nx.edge_dfs(genealogy,
                    idx, orientation='reverse'))[0][0])
                

                # Siblings
                for i in list(nx.dfs_edges(genealogy,parent_id[0])):
                    if(i[0] == parent_id[0]) & (i[1] != idx):
                        sibling_id.append(i[1])
                

                # All relevant ids
                nodes = parent_id + sibling_id + child_id
                
                # if it's not empty, let's get going
                if (len(predictionHints) > 0):
                    # create a plot for the time predictions (ignore which robot it came from)
                    c1 = alt.Chart(predictionHints).mark_circle(color = 'red', size = 150).transform_filter(
                        alt.datum.id == idx
                    ).encode(
                        alt.X('time:Q',scale=alt.Scale(domain=(0, 100))),
                        alt.Y('value:Q',scale=alt.Scale(domain=(0, 100)))
                    )

                    c2 = alt.Chart(predictionHints).mark_circle(color = 'blue', size = 100).transform_filter(
                        alt.FieldOneOfPredicate(field='id', oneOf = nodes)
                    ).encode(
                        alt.X('time:Q',scale=alt.Scale(domain=(0, 100))),
                        alt.Y('value:Q',scale=alt.Scale(domain=(0, 100))), 
                        tooltip = ['id:Q']
                    )

                    c3 = alt.Chart(predictionHints).mark_circle(color = 'black', opacity = 0.1, size = 50).transform_filter(
                        {'not': alt.FieldOneOfPredicate(field='id', oneOf = nodes+[idx])}
                    ).encode(
                        alt.X('time:Q',scale=alt.Scale(domain=(0, 100))),
                        alt.Y('value:Q',scale=alt.Scale(domain=(0, 100)))
                    )

                    c4 = alt.Chart(robotInfo[robotInfo['id'] == idx]).mark_rule(size = 4, color="green").encode(
                        x = alt.X('expires:Q')
                    )

                    expireTime = robotInfo[robotInfo['id'] == idx]['expires'].values[0]
                    temp_line = pd.DataFrame({'time':[expireTime for i in range(101)], 'value':[i for i in range(101)]})
                    c5 = alt.Chart(temp_line).mark_circle(size = 50, opacity = 0).encode(
                            x = alt.X('time:Q'),
                            y = alt.Y('value:Q'),
                            tooltip = ['value:Q']
                    )

                    VisSub = (c1 + c2 + c3 + c4 + c5).properties(
                        width = 200, height = 200, title = 'Robot ' + str(idx) + ', Expiration time ' + str(expireTime)
                    )

                    # Create the bar plot for each idx contain the information of productivity of its child, parents
                    ## sibling robots(nodes) in a single row 
                    ## idx: the id of the robot
                    ## nodes: a list contain the id of child, parents and siblings robots based on idx 
                    VisSub2 = productivity_graph(robotInfo, nodes)

                    VisSubto = VisSub & VisSub2
                    if count == 0:
                        Vis1 = VisSubto
                    else:
                        Vis1 = (Vis1|VisSubto)

                    count += 1
        
            if(count >= 5):
                break
            
        Vis = Vis1 
        title1.write('#### Robot Friendship Game')
        predVis.write(Vis)

        ###----------Create the popularity plot for the robots-------####
        # get the robot data  
        robots = game.getRobotInfo()
        
        # Add the expiration time and current team info to popularity:
        # ## merge with the robots info
        popularity = pd.DataFrame.from_dict(node, orient="index")
        popularity.reset_index(inplace=True)
        popularity.rename(columns={"index":"id", 0:"popularity"}, inplace=True)
            
        robots_info = robots[["id","expires"]]
        pop_expires = robots_info.merge(popularity, "inner", on="id")

        # remove the expired robots from dataset
        pop_expires = pop_expires[pop_expires.expires >= currentTimeGame]
        # only keep the top 10 robot. if there are multiple robots rank 10, keep all of them.
        pop_expires.sort_values(by=["popularity","id"], ascending=[False, True], inplace=True)
        pop_expires.reset_index(drop=True, inplace=True)
        popularit_rank10 = pop_expires.loc[9,"popularity"]
        pop_expires = pop_expires[pop_expires.popularity >= popularit_rank10]
        

        pop = alt.Chart(pop_expires).mark_bar(
            size = 15
        ).encode(
            x = alt.X("popularity:Q", 
                        axis= alt.Axis(orient="top", tickSize=0)),
            y = alt.Y("id:N",
                        sort = alt.EncodingSortField(field = "popularity", 
                                    order="descending"),
                        axis = alt.Axis(tickSize=0, titleAngle=0)),
            color = alt.Color("expires:Q",
                                scale=alt.Scale(scheme="reds",
                                            domain=[min(pop_expires.expires), 
                                                    max(pop_expires.expires)]),
                                sort = "descending"),
            opacity = alt.Opacity("expires:Q",
                        scale = alt.Scale(domain=[min(pop_expires.expires), 
                                                    max(pop_expires.expires)]),
                        sort="descending"),
            tooltip = ["popularity", "expires:Q"]
        ).properties(title="Popularity & Expiration Time")

        pop_plt = alt.layer(pop).configure_title(
                fontSize=14,
                anchor='start',
                dx = 20
            ).configure_axisY(
                titleAlign = "right"
            )
        
        partVis.write(pop_plt)
        ###----------Create the small multiples to show the relationship between attributes and productivity-------####

        quantProps = ['Astrogation Buffer Length','InfoCore Size',
            'AutoTerrain Tread Count','Polarity Sinks',
            'Cranial Uplink Bandwidth','Repulsorlift Motor HP',
            'Sonoreceptors']
        cateProps = ['Arakyd Vocabulator Model', 'Axial Piston Model',
                         'Nanochip Model']

        # update the hints
        game.getHints()

        # create a dataframe for the part hints
        allPartHints = pd.DataFrame(game.getAllPartHints()) # column, id, value
        robotInfo = pd.DataFrame(game.getRobotInfo()) # id, name, expires, winner, Productivity, bets
        
        if (len(allPartHints) > 0):
            
            # hacker will give duplicated information
            allPartHints_wide = pd.pivot(allPartHints.drop_duplicates(), index='id', 
                                         columns='column', values='value')
            
            # store and update all parts information
            df_allParts = pd.DataFrame(np.nan, index=[i for i in range(100)],
                                       columns=quantProps+cateProps)
            df_allParts = df_allParts.combine_first(allPartHints_wide)
            
            df_allParts_with_id = df_allParts.reset_index().rename(columns={'index': 'id'})
            df_prod_parts = df_allParts_with_id.merge(robotInfo, on='id', how='left')
            
            # all parts are object dtype
            for var in quantProps:
                df_prod_parts[var] = pd.to_numeric(df_prod_parts[var],errors = 'coerce')   
            for var in cateProps:
                df_prod_parts[var] = df_prod_parts[var].astype(pd.StringDtype())
            
            df_prod_parts[quantProps].apply(lambda x: pd.Series.round(x, 2))
            df_prod_parts.Productivity = df_prod_parts.Productivity.round(2)
            
            # small multiples
            selectLegend = alt.selection_multi(fields=['winner'], bind='legend')
            selecId = alt.selection_single(on='mouseover', empty='none') 
            sizeCondition = alt.condition(selecId, alt.SizeValue(200), alt.SizeValue(60)) 
            opacityCondition = alt.condition(selectLegend, alt.value(0.5), alt.value(0))
            
            hline = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(size=1).encode(y=alt.Y('y:Q', title=''))
            
            scat_all = []
            for part in (quantProps+cateProps):
                scat = alt.Chart(df_prod_parts[['id','winner','Productivity',part]].dropna()
                ).mark_circle(opacity=0.5).encode(
                    x = part,
                    y= alt.Y('Productivity:Q', title='Productivity'),
                    tooltip = ['id:N', 'Productivity:Q'],
                    color = 'winner:N',
                    size = sizeCondition,
                    opacity=opacityCondition  
                ).add_selection(selecId, selectLegend).properties(
                    height=200, width=200
                ).interactive()
                scat_each = (scat+hline)
                scat_all.append(scat_each)
            
            prod_vis = ((scat_all[0]|scat_all[1]|scat_all[2]|scat_all[3])&
                        (scat_all[4]|scat_all[5]|scat_all[6])&
                        (scat_all[7]|scat_all[8]|scat_all[9]))
            
            title2.write('#### The relationship between productivity and attributes')
            scatVis.write(prod_vis)

