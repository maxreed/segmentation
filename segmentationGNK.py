# doing song segmentation with a gaussian novelty filter.

import numpy as np
import scipy.io.wavfile as wav
# this needs to be here so matplotlib doesn't freak out
import matplotlib 
matplotlib.use('Agg') 


def sinebell(lengthWindow):
    """
    window = sinebell(lengthWindow)
    
    Computes a "sinebell" window function of length L=lengthWindow
    
    The formula is:
        window(t) = sin(pi * t / L), t = 0..L-1
    """
    window = np.sin((np.pi * (np.arange(lengthWindow))) \
                    / (1.0 * lengthWindow))
    return window

def stft(data, window=sinebell(2048), hopsize=256.0, nfft=2048.0, \
         fs=44100.0):
    """
    X, F, N = stft(data, window=sinebell(2048), hopsize=1024.0,
                   nfft=2048.0, fs=44100)
                   
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
        data                  : one-dimensional time-series to be
                                analyzed
        window=sinebell(2048) : analysis window
        hopsize=1024.0        : hopsize for the analysis
        nfft=2048.0           : number of points for the Fourier
                                computation (the user has to provide an
                                even number)
        fs=44100.0            : sampling rate of the signal
        
    Outputs:
        X                     : STFT of data
        F                     : values of frequencies at each Fourier
                                bins
        N                     : central time at the middle of each
                                analysis window
    """
    
    # window defines the size of the analysis windows
    lengthWindow = window.size
    
    # !!! adding zeros to the beginning of data, such that the first
    # window is centered on the first sample of data
    data = np.concatenate((np.zeros(lengthWindow / 2.0),data))          
    lengthData = data.size
    
    # adding one window for the last frame (same reason as for the
    # first frame)
    numberFrames = np.ceil((lengthData - lengthWindow) / hopsize \
                           + 1) + 1  
    newLengthData = (numberFrames - 1) * hopsize + lengthWindow
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data, np.zeros([newLengthData - lengthData])))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an
    # even number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2.0 + 1
    
    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)
    
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameToProcess = window * data[beginFrame:endFrame]
        STFT[:,int(n)] = np.fft.rfft(frameToProcess, int(nfft));
        
    F = np.arange(numberFrequencies) / nfft * fs
    N = np.arange(numberFrames) * hopsize / fs
    
    return STFT, F, N

def furtherCuts(start,stop,noveltyScores):
    # if there are less than 4 seconds in the clip being examined, return that no transitions were found.
    if stop-start<89:
        return []
    # find the average novelty score in the section of novelty scores being examined.
    averageNovScore = np.average(noveltyScores)
    # make the novelty scores a list so it can be easily sorted and indexed and whatnot.
    noveltyScores = list(noveltyScores)
    # initialize the list of transitions in this section.
    transitions = []
    # find the initial greatest novelty score.
    initialMaxNovelty = max(noveltyScores)
    # make a boundary novelty scores. transitions with scores above this are considered valid transitions.
    boundaryScore = (averageNovScore*3 + initialMaxNovelty)/4
    # find the initial maximum novelty score's locations
    initialMaxNoveltyLoc = noveltyScores.index(initialMaxNovelty)
    # if the initial maximum is within 2 seconds of the start or end, reduce the section of novelty scores being examined to exclude it.
    if initialMaxNoveltyLoc<44:
        return furtherCuts(start+44,stop,noveltyScores[44:])
    elif initialMaxNoveltyLoc>(stop-start-44):
        return furtherCuts(start,stop-44,noveltyScores[:-44])
    # loop until all valid transitions have been found
    continueLoop = True
    while(continueLoop):
        # find the greatest novelty score
        maxNovelty = max(noveltyScores)
        # find its location
        maxNoveltyLoc = noveltyScores.index(maxNovelty)
        # initialize the value for the distance to the nearest transition in the segment being examined
        closestTransition = 10**8
        # find the location of the nearest transition
        for i in np.arange(len(transitions)):
            if abs(transitions[i]-maxNoveltyLoc)<closestTransition:
                closestTransition = abs(transitions[i]-maxNoveltyLoc)
        # zero this maximum and its immediate vicinity
        for j in np.arange(-5,6):
            if j>=0 and j<(stop-start):
                noveltyScores[maxNoveltyLoc+j] = 0
        if maxNovelty > boundaryScore:
            if closestTransition>88:
                if maxNoveltyLoc>44 and maxNoveltyLoc<(stop-start-44):
                    # if the greatest novelty score is sufficiently large, at least 4 seconds from the next nearest transition, and at least 2 seconds from the start or end, it is a valid transition.
                    transitions.append(maxNoveltyLoc)
        else:
            # if the maximum is not sufficiently large, it is not a valid transition and all potentially valid transitions have been found.
            continueLoop = False
        # adjust the transitions to account for the fact that this section is part way into the song.
        for i in np.arange(len(transitions)):
            transitions[i] = transitions[i]+start
        # print and return the transitions found.
        print "Transitions from further cuts: " + str(transitions)
        return transitions

def findPreciseTransition(song,fs,data):
    # make the two data channels be one channel
    dataMod = (data[:,0]+data[:,1])/2
    # perform a fourier transform on the data
    X, F, N = stft(dataMod)
    # ft data is complex, so make it real
    SX = np.maximum(np.abs(X) ** 2, 10**-8)
    # normalize the ft data.
    for i in np.arange(SX.shape[1]):
        min_in_sx = min(SX[:,i])
        SX[:,i] = SX[:,i] - min_in_sx
        max_in_sx = max(SX[:,i])
        if max_in_sx==0:
            max_in_sx = 10**-8
        SX[:,i] = SX[:,i]/max_in_sx
    # we can throw out the high frequencies. we can also average every 4 time frames into 1.
    a = len(N)/4+1
    SX_smoothed = np.zeros((200, a))
    for i in np.arange(200):
        for j in np.arange(2, len(N)-2, 4):
            SX_smoothed[i,j/4] = sum(SX[i,j-4:j+4])/9
    # take note of the new number of frames
    numFrames = SX_smoothed.shape[1]
    
    # number of time frames to look at to determine the score at a point.
    numFramesExamined = 12
    
    # initialize the array for scores. the lower a score at [a,b], the more similar areas a and b are. (score[a,b] should be 0 when they're identical.)
    scores = np.zeros(((numFrames-numFramesExamined)/2+1,(numFrames-numFramesExamined)/2+1))
    
    # more compressing of data is happening, so skip every other time frame.
    for i in np.arange(0,numFrames-numFramesExamined,2):
        # only the values around the diagonal need to be calculated.
        for j in np.arange(i,min(numFrames-numFramesExamined, i+50),2):
            scores[i/2,j/2] = sum(sum(abs(SX_smoothed[:,i:i+numFramesExamined]-SX_smoothed[:,j:j+numFramesExamined])))
            # this can be done because the data is symmetric across the diagonal.
            scores[j/2,i/2] = scores[i/2,j/2]

    # the scores are gonna get smoothed more now cause they're relatively noisy.
    smoothScores = np.zeros(((numFrames-numFramesExamined)/2+1,(numFrames-numFramesExamined)/2+1))
    for i in np.arange(2,(numFrames-numFramesExamined)/2-1):
        for j in np.arange(i,min((numFrames-numFramesExamined)/2-1, i+20)):
            smoothScores[i,j] = sum(sum(scores[i-2:i+2,j-2:j+2]))/25
            # again, the scores are symmetric across the diagonal.
            smoothScores[j,i] = smoothScores[i,j]
            
    # normalizing the smoothedScores (making it vary from 0 to 1)
    minSmoothScores = np.min(smoothScores)
    smoothScores = smoothScores - minSmoothScores
    maxSmoothScores = np.max(smoothScores)
    smoothScores = smoothScores / maxSmoothScores
    
    # initialize the array holding the novelty scores.
    noveltyScores = np.zeros(smoothScores.shape[0])
    
    # make the gaussian novelty kernel
    gnk = np.zeros((31,31))
    for i in np.arange(31):
        for j in np.arange(31):
            gnk[i][j] = np.exp(-(np.sqrt(((i-15)**2 + (j-15)**2)))**2/162)
    gnk[:15,:15] = - gnk[:15,:15]
    gnk[16:,16:] = - gnk[16:,16:]
    gnk[15,:]=0
    gnk[:,15]=0
    
    for i in np.arange(31):
        gnk[i,i]=0
        gnk[30-i,i]=0
    
    # calculate novelty scores. a sections with absolutely no change, should get a novelty score of 0, a section with lots of change should get a high score.
    for i in np.arange(16,smoothScores.shape[0]-16):
        noveltyScores[i] = abs(np.sum(np.multiply(gnk,smoothScores[i-15:i+16,i-15:i+16])))
    
    # calculate the average novelty score
    averageNovScore = np.average(noveltyScores)
    # make novelty scores a list so it can be easily sorted and indexed and whatnot
    noveltyScores = list(noveltyScores)
    # initialize the list of transitions between different segments
    transitions = [0]
    # find the highest novelty score
    initialMaxNovelty = max(noveltyScores)
    # calculate a boundary for novelty scores. any score above this will be considered a valid transition.
    boundaryScore = (averageNovScore + initialMaxNovelty)/2
    # print stuff
    print averageNovScore
    print initialMaxNovelty
    print boundaryScore
    # continue looping
    continueLoop = True
    while(continueLoop):
        # find the greatest novelty score
        maxNovelty = max(noveltyScores)
        # find its location
        maxNoveltyLoc = noveltyScores.index(maxNovelty)
        # initialize the distance to the nearest transition between segments.
        closestTransition = 10**8
        # find the distance from this maximum to the nearest existing transition between segments (do not include 0 as a transition, that's just the song's start).
        for i in np.arange(1,len(transitions)):
            if abs(transitions[i]-maxNoveltyLoc)<closestTransition:
                closestTransition = abs(transitions[i]-maxNoveltyLoc)
        # make the maximum novelty score and its immediate vicinity become 0.
        for j in np.arange(-5,6):
            noveltyScores[maxNoveltyLoc+j] = 0
        if maxNovelty > boundaryScore:
            if closestTransition>88:
                # if the novelty score is sufficiently high and at least four seconds away from other transitions, it is a valid transition between segments.
                transitions.append(maxNoveltyLoc)
        else:
            # if the novelty score was not high enough, then all sufficiently high novelty scores have been found. stop looping.
            continueLoop = False
    # append a transition for the spot corresponding to the song's end.
    transitions.append(len(dataMod)/(8*256))
    # sort the transitions.
    transitions.sort()
    print "Initial transitions: " + str(transitions)
    
    # this next bit assures there is at least 1 transition every 16 seconds (unless there end up being absolutely no good spots to transition within a length of time greater than 16 seconds, but that rarely happens).
    counter = 0
    # go through the list of transitions
    while counter<(len(transitions)-1):
        if (transitions[counter+1]-transitions[counter]>(16*fs/(256*8))):
            # if there are more than 16 seconds between transitons, find new transitions between those two transitions
            newTransitions = furtherCuts(transitions[counter]+11,transitions[counter+1]-11,noveltyScores[transitions[counter]+11:transitions[counter+1]-11])
            if len(newTransitions)==0:
                # if no new transitons were found, just keep going.
                counter += 1
            else:
                # if new transitions were found, add them to the list of transitions, sort the transitions, and restart going through the list of transitions.
                counter=0
                for j in np.arange(len(newTransitions)):
                    transitions.append(newTransitions[j])
                transitions.sort()
        else:
            # if there are less than 16 seconds between two given transitons, move on.
            counter += 1
    print "Updated transitions: " + str(transitions)
    
    # change the transitions to correspond to points in the song data.
    for i in np.arange(len(transitions)):
        transitions[i]=transitions[i]*256*8
    
    # write the segments.
    for i in np.arange(len(transitions)-1):
        wav.write("music_sections/" + song[6:-4] + str(i) + ".wav", fs, data[transitions[i]:transitions[i+1],:])

import time

# shitty hardcoding of specific songs to semgent  ¯\_(ツ)_/¯
# read in a song
start = time.time()
fs, data = wav.read("music/ForeverYoung.wav")
findPreciseTransition("music/ForeverYoung.wav",fs,data)
stop = time.time()
print "Forever Young: " + str(stop-start)
start = time.time()
fs, data = wav.read("music/radioactive.wav")
findPreciseTransition("music/radioactive.wav",fs,data)
stop = time.time()
print "Radioactive: " + str(stop-start)
start = time.time()
fs, data = wav.read("music/numa_numa.wav")
findPreciseTransition("music/numa_numa.wav",fs,data)
stop = time.time()
print "Numa Numa: " + str(stop-start)
start = time.time()
fs, data = wav.read("music/call_on_me.wav")
findPreciseTransition("music/call_on_me.wav",fs,data)
stop = time.time()
print "Call On Me: " + str(stop-start)
start = time.time()
fs, data = wav.read("music/worth_it.wav")
findPreciseTransition("music/worth_it.wav",fs,data)
stop = time.time()
print "Worth It: " + str(stop-start)
start = time.time()
fs, data = wav.read("music/my_heart_will_go_on.wav")
findPreciseTransition("music/my_heart_will_go_on.wav",fs,data)
stop = time.time()
print "My Heart Will Go On: " + str(stop-start)
start = time.time()
fs, data = wav.read("music/let_her_go.wav")
findPreciseTransition("music/let_her_go.wav",fs,data)
stop = time.time()
print "Let Her Go: " + str(stop-start)
