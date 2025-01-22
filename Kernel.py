import time

import numpy as np
import pandas as pd

import os, queue, sys
from message.Message import MessageType

from util.util import log_print


class Kernel:

    def __init__(self, kernel_name, random_state=None):
        # kernel_name for human readability
        self.name = kernel_name
        self.random_state = random_state

        if not random_state:
            raise ValueError("A valid, seeded np.random.RandomState object is required " +
                             "for the Kernel", self.name)
            sys.exit()

        # Single message queue, sorted by delivery timestamp ascending
        self.messages = queue.PriorityQueue()

        # currentTime is None until all agents have completed their kernelStarting() event
        # this is a pd.Timestamp containing a date
        self.currentTime = None

        # Timestamp that the Kernel was created. Primarily used for creating a unique log directory for this run
        # also used to print some elapsed time and messages per second statistics
        self.kernelWallClockStart = pd.Timestamp('now')

        # TODO: this belongs in the finance part, probably not here...
        self.meanResultByAgentType = {}
        self.agentCountByType = {}

        # The Kernel maintains a summary log, that agents can write to
        # that should be information centralized for quick access by a separate statistics summarizer
        # verbose event logging should be written only to agent's personal logs
        # this is for things like "final position value"
        self.summaryLog = []

        log_print("Kernel initialized: {}", self.name)

    # This is called to actually start the simulation, once all agents have been configured
    def runner(self, agents=[], startTime=None, stopTime=None,
               num_simulations=1, defaultComputationDelay=1,
               defaultLatency=1, agentLatency=None, latencyNoise=[1.0],
               agentLatencyModel=None, skip_log=False,
               seed=None, oracle=None, log_dir=None):

        # agents must be a list of Agents to simulate,
        #        based on class agent.Agent
        self.agents = agents

        # Simulation custom state, stored in a free-format dictionary. Allows for config files that drive multiple
        # simulations, or the ability to generate special logs after the simulation, to get the desired output, without
        # writing special case code in the Kernel.
        # each agent's state should be handled with the provided updateAgentState() method.
        self.custom_state = {}

        # Simulation start and stop times (the first and last timestamps in the simulation, unrelated to exchange open/close, etc)
        self.startTime = startTime
        self.stopTime = stopTime

        # global seed, not used for anything agent related.
        self.seed = seed

        # Should the Kernel skip writing Agent logs?
        self.skip_log = skip_log

        # The data oracle for this simulation (if required).
        self.oracle = oracle

        # If no log directory specified, use the initial wallclock time.
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = str(int(self.kernelWallClockStart.timestamp()))

        # The Kernel maintains a current time for each agent, to allow
        # for the simulation of each agent's computation delay. The agent's time
        # is advanced forward on each wake up (see below) and the
        # agent will not be able to receive new messages/wake up until
        # the global time reaches the agent's time.
        # (ie it cannot take an action again until it is "in the future")

        # This also nicely enforces that agents can not take actions before the simulation start time.
        self.agentCurrentTimes = [self.startTime] * len(agents)

        # agentComputationDelays, in nanoseconds, starts with a default from the config,
        # and can be changed by any agent (limited to itself) at any time. It represents the
        # time penalty applied every time an agent wakes up (either wakeup or recvMsg). The
        # penalty is applied _after_ the agent takes it's action, before it can act again.
        # TODO: This may be changed to pd.Timedelta objects someday.
        self.agentComputationDelays = [defaultComputationDelay] * len(agents)

        # if an agentLatencyModel is defined, it will be used in place of the
        # older, non-model based properties.
        self.agentLatencyModel = agentLatencyModel

        # If agentLatencyModel is not defined, the older parameters:
        # agentLatency (or defaultLatency) and latencyNoise should be specified.
        # these should be considered deprecated and removed in the future.

        # if agentLatency is not defined, define it using defaultLatency.
        # This matrix defines communication latency between each agent pair.
        if agentLatency is None:
            self.agentLatency = [[defaultLatency] * len(agents)] * len(agents)
        else:
            self.agentLatency = agentLatency

        # There is a latency noise model, intended to be a single sided
        # distribution, peaked at zero. By default there is no noise
        # (100% chance of adding zero ns of extra latency). Format is a list,
        # list index = extra ns latency, value = probability of this latency.
        self.latencyNoise = latencyNoise

        # The Kernel maintains an accumulated additional delay parameter
        # for the current agent. This is applied to every message sent
        # as well as when returning from wakeup/receive message, in addition
        # to the agent's standard computation delay. However, it is never carried
        # over to future wakeup/receive message calls. It is useful for
        # interleaving the sending of messages.
        self.currentAgentAdditionalDelay = 0

        log_print("Kernel started: {}", self.name)
        log_print("Simulation started!")

        # Note that num_simulations has not really been used or tested
        # for anything yet. Instead, we have been running multiple simulations
        # using crude parallelization from shell scripts.
        for sim in range(num_simulations):
            log_print("Starting sim {}", sim)

            # Notification of Kernel initialization (agents should not attempt to
            # communicate with other agents, as the order is unknown). Agents
            # should initialize any internal resources they may need during agent.kernelStarting()
            # to communicate with other agents. The Kernel passes a self-reference for the agent to retain,
            # so they may communicate with the Kernel in the future (as it has no agent ID).
            log_print("\n--- Agent.kernelInitializing() ---")
            for agent in self.agents:
                agent.kernelInitializing(self)

            # Notification of Kernel start (agents may set up communication
            # or references to other agents, as all agents are now
            # guaranteed to exist). Agents should get references to
            # the agents they need to operate normally (exchanges, brokers, subscription services...). Note that we usually
            # do not (and should not) allow an agent to get direct references to other agents, like exchanges, as they could
            # bypass the Kernel, and thus the simulation "physics", by directly and immediately sending messages
            # or doing illicit direct inspection of other agents states. Agents should instead get
            # Agent IDs for other agents, and only communicate to them via the
            # Kernel. Direct references to non-agent utility objects are acceptable
            # (eg, oracles).
            log_print("\n--- Agent.kernelStarting() ---")
            for agent in self.agents:
                agent.kernelStarting(self.startTime)

            # Set Kernel to its startTime.
            self.currentTime = self.startTime
            log_print("\n--- Kernel Clock started ---")
            log_print("Kernel.currentTime is now {}", self.currentTime)

            # Begin processing the event queue.
            log_print("\n--- Kernel Event Queue begins ---")
            log_print("Kernel will start processing messages.  Queue length: {}", len(self.messages.queue))

            # track the wallclock start time and total messages for the event queue for statistics at the end.
            eventQueueWallClockStart = pd.Timestamp('now')
            ttl_messages = 0

            # Process messages until there are no messages left (at which point there will never be more, because agents only
            # "wake up" in response to messages) or until the Kernel stop time is reached.
            while not self.messages.empty() and self.currentTime and (self.currentTime <= self.stopTime):
                # Get the next message from the timestamp order (delivery time) and extract it.
                self.currentTime, event = self.messages.get()
                msg_recipient, msg_type, msg = event

                # periodically print the simulation time, and total messages, even if muted.
                if ttl_messages % 100000 == 0:
                    print("\n--- Simulation time: {}, messages processed: {}, wallclock elapsed: {} ---\n".format(
                        self.fmtTime(self.currentTime), ttl_messages, pd.Timestamp('now') - eventQueueWallClockStart))

                log_print("\n--- Kernel Event Queue pop ---")
                log_print("Kernel handling {} message for agent {} at time {}",
                          msg_type, msg_recipient, self.fmtTime(self.currentTime))

                ttl_messages += 1

                # Between messages, always reset the currentAgentAdditionalDelay.
                self.currentAgentAdditionalDelay = 0

                # Dispatch the message to the agent.
                if msg_type == MessageType.WAKEUP:

                    # Who requested this wake up call?
                    agent = msg_recipient

                    # Test if the agent is already in the future. If it is,
                    # defer the wake up until the agent can act again.
                    if self.agentCurrentTimes[agent] > self.currentTime:
                        # Put the wakeup call back into the PQ with the new time.
                        self.messages.put((self.agentCurrentTimes[agent],
                                           (msg_recipient, msg_type, msg)))
                        log_print("Agent in future: wakeup requeued for {}",
                                  self.fmtTime(self.agentCurrentTimes[agent]))
                        continue

                    # Set the agent's current time to the global current time in order to start
                    # processing.
                    self.agentCurrentTimes[agent] = self.currentTime

                    # Wake up the agent.
                    agents[agent].wakeup(self.currentTime)

                    # Delay the agent by the agent's computation delay, plus any transient additional delay.
                    self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] +
                                                                  self.currentAgentAdditionalDelay)

                    log_print("After wakeup return, agent {} delayed from {} to {}",
                              agent, self.fmtTime(self.currentTime), self.fmtTime(self.agentCurrentTimes[agent]))

                elif msg_type == MessageType.MESSAGE:

                    # Who is receiving this message?
                    agent = msg_recipient

                    # Test if the agent is already in the future. If it is,
                    # defer the message until the agent can act again.
                    if self.agentCurrentTimes[agent] > self.currentTime:
                        # Put the message back into the PQ with the new time.
                        self.messages.put((self.agentCurrentTimes[agent],
                                           (msg_recipient, msg_type, msg)))
                        log_print("Agent in future: message requeued for {}",
                                  self.fmtTime(self.agentCurrentTimes[agent]))
                        continue

                    # Set the agent's current time to the global current time in order to start
                    # processing.
                    self.agentCurrentTimes[agent] = self.currentTime
                    # by cx 2024.05.28
                    # time.sleep(1)
                    # Pass the message.
                    agents[agent].receiveMessage(self.currentTime, msg)

                    # Delay the agent by the agent's computation delay, plus any transient additional delay.
                    self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] +
                                                                  self.currentAgentAdditionalDelay)

                    log_print("After receiveMessage return, agent {} delayed from {} to {}",
                              agent, self.fmtTime(self.currentTime), self.fmtTime(self.agentCurrentTimes[agent]))

                else:
                    raise ValueError("Unknown message type found in queue",
                                     "currentTime:", self.currentTime,
                                     "messageType:", self.msg.type)


            if self.messages.empty():
                log_print("\n--- Kernel Event Queue empty ---")

            if self.currentTime and (self.currentTime > self.stopTime):
                log_print("\n--- Kernel Stop Time surpassed ---")

            # record the wall clock stop time and elapsed time for statistics at the end.
            eventQueueWallClockStop = pd.Timestamp('now')

            eventQueueWallClockElapsed = eventQueueWallClockStop - eventQueueWallClockStart

            # Notification of Kernel stop (agents can communicate with
            # other agents, as all agents are still guaranteed to exist).
            # Agents should not destroy resources that they may need to respond to
            # final communications from other agents.
            log_print("\n--- Agent.kernelStopping() ---")
            for agent in agents:
                agent.kernelStopping()

            # Notification of Kernel terminate (agents should not
            # attempt to communicate with other agents, as the termination order
            # is unknown). Agents should clean up any resources they used, as
            # the simulation program may not actually terminate if num_simulations > 1.
            log_print("\n--- Agent.kernelTerminating() ---")
            for agent in agents:
                agent.kernelTerminating()

            print("Event Queue elapsed: {}, messages: {}, messages per second: {:0.1f}".format(
                eventQueueWallClockElapsed, ttl_messages,
                ttl_messages / (eventQueueWallClockElapsed / (np.timedelta64(1, 's')))))
            log_print("Ending sim {}", sim)

        # The Kernel adds some custom state results for all simulations,
        # which the config file can use, print, log, or discard.
        self.custom_state['kernel_event_queue_elapsed_wallclock'] = eventQueueWallClockElapsed
        self.custom_state['kernel_slowest_agent_finish_time'] = max(self.agentCurrentTimes)

        # Agents will request the Kernel to serialize their agent logs, usually
        # during kernelTerminating, but the Kernel must write out its summary
        # log itself.
        self.writeSummaryLog()

        # this probably should go somewhere else, as it's explicitly finance related, but
        # it's convenient to have a quick results summary for now.
        print("Mean ending value by agent type:")
        for a in self.meanResultByAgentType:
            value = self.meanResultByAgentType[a]
            count = self.agentCountByType[a]
            print("{}: {:d}".format(a, int(round(value / count))))

        print("Simulation ending!")

        return self.custom_state

    def sendMessage(self, sender=None, recipient=None, msg=None, delay=0, tag=None):
        # """
        # Args:
        #     - **sender (str):** ID of the agent sending the message. This is a required argument.
        #     - **recipient (str):** ID of the agent receiving the message. This is a required argument.
        #     - **msg (Message):** Message to send, must be an instance of class message.Message. This is a required argument.
        #     - **delay (pd.Timedelta, optional):** Extra delay requested by the agent, for indicating parallel pipeline
        #       processing delays. Defaults to 0, indicating no extra delay.
        #     - **tag (str, optional):** A tag for recording custom state for tracking message latencies.
        #
        # Returns:
        #
        # """

        # Called by an agent to send a message to another agent. The Kernel
        # provides its own currentTime (ie "now") to prevent possible
        # abuses by agents. The Kernel will handle the computation delay penalty
        # and/or network latencies. Messages must derive from the message.Message class.
        # The optional delay argument represents a request by the agent for extra
        # latency (beyond the Kernel's enforced computation + latency) to represent
        # parallel pipeline processing latency (which should delay the message transfer
        # but not make the agent "busy" and unable to respond to new messages).

        if sender is None:
            raise ValueError("sendMessage() called without valid sender ID",
                             "sender:", sender, "recipient:", recipient,
                             "msg:", msg)

        if recipient is None:
            raise ValueError("sendMessage() called without valid recipient ID",
                             "sender:", sender, "recipient:", recipient,
                             "msg:", msg)

        if msg is None:
            raise ValueError("sendMessage() called with message == None",
                             "sender:", sender, "recipient:", recipient,
                             "msg:", msg)

        # Apply the agents current computation delay to effectively "send" the message
        # at the _end_ of the agents current compute cycle, at the point where it's done "thinking".
        # Note that sending multiple messages on a single wakeup will transmit them all at the same time,
        # at the end of the computation. To avoid this, use Agent.delay() to accumulate
        # a temporary delay (current loop only), which will also interleave messages.

        # The optional pipeline delay parameter _does_ move the sent time forward, as it
        # represents "thinking" time before the message is sent. We don't make much use of this currently,
        # but it may be important later.

        # This means that the message delay (before latency) is the agent's standard computation delay
        # plus any accumulated delay for this wakeup loop, plus any one-time requested delay for this specific message
        # only.
        # delay = 1000000000
        sentTime = self.currentTime + pd.Timedelta(self.agentComputationDelays[sender] +
                                                   self.currentAgentAdditionalDelay + delay)

        # sentTime = self.currentTime + pd.Timedelta(
        #     seconds=self.agentComputationDelays[sender] + self.currentAgentAdditionalDelay) + delay

        # Apply the communication delay as determined by the agentLatencyModel (if defined) or
        # the agentLatency matrix [sender][recipient]
        if self.agentLatencyModel is not None:
            latency = self.agentLatencyModel.get_latency(sender_id=sender, recipient_id=recipient)
            deliverAt = sentTime + pd.Timedelta(latency)

            # record the tagged latency.
            if tag: self.custom_state[tag] = self.custom_state.get(tag, pd.Timedelta(0)) + pd.Timedelta(latency)

            log_print(
                "Kernel applied latency {}, accumulated delay {}, one-time delay {} on sendMessage from: {} to {}, scheduled for {}",
                latency, self.currentAgentAdditionalDelay, delay, self.agents[sender].name, self.agents[recipient].name,
                self.fmtTime(deliverAt))
        else:
            latency = self.agentLatency[sender][recipient]
            noise = self.random_state.choice(len(self.latencyNoise), 1, self.latencyNoise)[0]
            deliverAt = sentTime + pd.Timedelta(latency + noise)
            log_print(
                "Kernel applied latency {}, noise {}, accumulated delay {}, one-time delay {} on sendMessage from: {} to {}, scheduled for {}",
                latency, noise, self.currentAgentAdditionalDelay, delay, self.agents[sender].name,
                self.agents[recipient].name,
                self.fmtTime(deliverAt))


        # Finally put the message into the queue, with priority == delivery time.
        self.messages.put((deliverAt, (recipient, MessageType.MESSAGE, msg)))

        log_print("Sent time: {}, current time {}, computation delay {}", sentTime, self.currentTime,
                  self.agentComputationDelays[sender])
        log_print("Message queued: {}", msg)

    def setWakeup(self, sender=None, requestedTime=None):
        # Called by an agent, to receive a "wakeup call" from the Kernel at some requested future time.
        # Defaults to the next possible timestamp. Wakeup time cannot be current or in the past.
        # Sender is required, and should be the ID of the agent initiating the call.
        # Agents are responsible for maintaining any required state; the Kernel does not
        # provide any parameters to the wakeup() call.

        if requestedTime is None:
            requestedTime = self.currentTime + pd.Timedelta(1)

        if sender is None:
            raise ValueError("setWakeup() called without valid sender ID",
                             "sender:", sender, "requestedTime:", requestedTime)

        if self.currentTime and (requestedTime < self.currentTime):
            raise ValueError("setWakeup() called with requested time not in future",
                             "currentTime:", self.currentTime,
                             "requestedTime:", requestedTime)

        log_print("Kernel adding wakeup for agent {} at time {}",
                  sender, self.fmtTime(requestedTime))

        self.messages.put((requestedTime,
                           (sender, MessageType.WAKEUP, None)))

    def getAgentComputeDelay(self, sender=None):
        # Allows an agent to query its current compute delay.
        return self.agentComputationDelays[sender]

    def setAgentComputeDelay(self, sender=None, requestedDelay=None):
        # Called by an agent to update its computation delay. This will not
        # initiate a global delay, nor will it initiate an
        # immediate delay for the agent. Rather it sets the new default delay for the calling
        # agent. The delay will be applied each time returning from wakeup
        # or recvMsg. Note that this delay _does_ apply to any messages
        # sent by the agent during the current wakeup loop (to simulate
        # messages popping at the end of it's "thinking" time).

        # Also note that we _do_ allow zero compute delay, but this should
        # really only be used for special or massively parallel agents.

        # requestedDelay should be in whole nanoseconds.
        if not type(requestedDelay) is int:
            raise ValueError("Requested computation delay must be whole nanoseconds.",
                             "requestedDelay:", requestedDelay)

        # requestedDelay must be non-negative.
        if not requestedDelay >= 0:
            raise ValueError("Requested computation delay must be non-negative nanoseconds.",
                             "requestedDelay:", requestedDelay)

        self.agentComputationDelays[sender] = requestedDelay

    def delayAgent(self, sender=None, additionalDelay=None):
        # Called by an agent to accumulate a temporary delay for the current wakeup loop.
        # This will apply a total delay to every message (on sendMessage),
        # and will modify the agents next available time period. These occur on top
        # of the agent's compute delay _but will not change it_. (ie the effect is transient)
        # Primarily for interleaving outgoing messages.

        # additionalDelay should be in whole nanoseconds.
        if not type(additionalDelay) is int:
            raise ValueError("Additional delay must be whole nanoseconds.",
                             "additionalDelay:", additionalDelay)

        # additionalDelay must be non-negative.
        if not additionalDelay >= 0:
            raise ValueError("Additional delay must be non-negative nanoseconds.",
                             "additionalDelay:", additionalDelay)

        self.currentAgentAdditionalDelay += additionalDelay

    def findAgentByType(self, type=None):
        # Called to request the ID of an arbitrary agent matching the class or base class passed as "type".
        # For example, any ExchangeAgent, or any NasdaqExchangeAgent.
        # This method is relatively expensive, and thus the result should be cached by the caller!

        for agent in self.agents:
            if isinstance(agent, type):
                return agent.id

    def writeLog(self, sender, dfLog, filename=None):
        # Called by any agent, usually before the end of simulation
        # kernel shutdown, to write any log dataframe to disk that it has
        # accumulated during simulation. Format is up to the agent, though changes
        # will require a specialized tool to read and parse the logs. The Kernel places
        # logs in a unique directory per run, with one filename per agent, also
        # determined by the Kernel using agent type, ID, etc.

        # It may not be good to put all these files in a directory if there are too
        # many agents. If there are too many agents, or if the logs are too
        # large, memory might be a problem. In that case we might have to take a
        # speed hit to incrementally write the logs.

        # If filename is not None, it will be used as the filename. Otherwise
        # Kernel will construct a filename based on the name of the Agent requesting the log archive.

        if self.skip_log: return

        path = os.path.join(".", "log", self.log_dir)

        if filename:
            file = "{}.bz2".format(filename)
        else:
            file = "{}.bz2".format(self.agents[sender].name.replace(" ", ""))

        if not os.path.exists(path):
            os.makedirs(path)

        dfLog.to_pickle(os.path.join(path, file), compression='bz2')

    def appendSummaryLog(self, sender, eventType, event):
        # We don't even include the timestamp, as this log is only for one time
        # summary reporting, like starting cash or ending cash.
        self.summaryLog.append({'AgentID': sender,
                                'AgentStrategy': self.agents[sender].type,
                                'EventType': eventType, 'Event': event})

    def writeSummaryLog(self):
        path = os.path.join(".", "log", self.log_dir)
        file = "summary_log.bz2"

        if not os.path.exists(path):
            os.makedirs(path)

        dfLog = pd.DataFrame(self.summaryLog)

        dfLog.to_pickle(os.path.join(path, file), compression='bz2')

    def updateAgentState(self, agent_id, state):
        # """
        #     Called by an agent that wants to replace the custom state in the dictionary that the Kernel returns at the
        #     end of the simulation.
        #     The Kernel returns this dictionary at the end of simulation. Shared state must be set directly,
        #     and agents should coordinate setting it non-destructively.
        #
        #     Note that an agent should never need to use this Kernel state dictionary to remember information
        #     about itself, but only to report back to the config file.
        # """

        if 'agent_state' not in self.custom_state: self.custom_state['agent_state'] = {}
        self.custom_state['agent_state'][agent_id] = state

    @staticmethod
    def fmtTime(simulationTime):
        # The Kernel class knows how to pretty print times. simulationTime is assumed
        # to be nanoseconds since midnight. Note this is a static method and can be
        # called on the class or an instance.

        # Try to only return a pd.Timestamp now.
        return (simulationTime)

        ns = simulationTime
        hr = int(ns / (1000000000 * 60 * 60))
        ns -= (hr * 1000000000 * 60 * 60)
        m = int(ns / (1000000000 * 60))
        ns -= (m * 1000000000 * 60)
        s = int(ns / 1000000000)
        ns = int(ns - (s * 1000000000))

        return "{:02d}:{:02d}:{:02d}.{:09d}".format(hr, m, s, ns)