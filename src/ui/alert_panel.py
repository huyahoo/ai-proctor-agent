from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal

class AlertPanel(QWidget):
    feedback_provided = pyqtSignal(dict, str) # event_data, feedback_type (confirmed_cheating, false_positive)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout().setContentsMargins(10, 10, 10, 10) # Padding
        self.layout().setSpacing(10)

        self.alert_title_label = QLabel("Anomaly Detection & AI Reasoning")
        self.alert_title_label.setStyleSheet("""
            font-weight: bold;
            font-size: 18px;
            color: #2c3e50; /* Dark blue/gray */
            margin-bottom: 10px;
        """)
        self.layout().addWidget(self.alert_title_label)

        # Scroll area for initial anomaly events
        self.initial_events_group = QFrame(self)
        self.initial_events_group.setLayout(QVBoxLayout())
        self.initial_events_group.layout().setContentsMargins(0,0,0,0)
        self.initial_events_group.layout().setSpacing(5)
        self.initial_events_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.initial_events_scroll_area = QScrollArea(self)
        self.initial_events_scroll_area.setWidgetResizable(True)
        self.initial_events_scroll_area.setWidget(self.initial_events_group)
        self.initial_events_scroll_area.setStyleSheet("border: 1px solid #ccc; border-radius: 5px;")

        self.layout().addWidget(QLabel("<b>Detected Anomalies (Click to Analyze):</b>"))
        self.layout().addWidget(self.initial_events_scroll_area)


        # AI Reasoning Display Area (Initially hidden or minimal)
        self.ai_reasoning_group = QFrame(self)
        self.ai_reasoning_group.setFrameShape(QFrame.Shape.StyledPanel)
        self.ai_reasoning_group.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 8px; padding: 15px;")
        self.ai_reasoning_layout = QVBoxLayout(self.ai_reasoning_group)
        self.ai_reasoning_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.layout().addWidget(QLabel("<b>AI Reasoning Process:</b>"))
        self.layout().addWidget(self.ai_reasoning_group)

        self._init_ai_reasoning_display()

        self.active_event_data = None # Store data for the event being displayed/feedbacked

    def _init_ai_reasoning_display(self):
        # Event Info
        self.current_event_label = QLabel("<i>No event selected for analysis.</i>")
        self.current_event_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        self.ai_reasoning_layout.addWidget(self.current_event_label)

        self.ai_reasoning_layout.addSpacing(10)

        # LLM Constraints Display
        self.llm_title = QLabel("<b>LLM Generated Constraint:</b>")
        self.llm_constraint_label = QLabel("Waiting for analysis...")
        self.llm_constraint_label.setWordWrap(True)
        self.llm_constraint_label.setStyleSheet("background-color: #ecf0f1; padding: 8px; border-left: 3px solid #3498db; border-radius: 4px;")
        self.ai_reasoning_layout.addWidget(self.llm_title)
        self.ai_reasoning_layout.addWidget(self.llm_constraint_label)
        self.ai_reasoning_layout.addSpacing(10)

        # VLM Analysis Display
        self.vlm_title = QLabel("<b>VLM Analysis & Decision:</b>")
        self.vlm_decision_label = QLabel("Waiting for analysis...")
        self.vlm_decision_label.setStyleSheet("font-weight: bold; font-size: 15px;")
        self.vlm_explanation_label = QLabel("")
        self.vlm_explanation_label.setWordWrap(True)
        self.vlm_explanation_label.setStyleSheet("background-color: #ecf0f1; padding: 8px; border-left: 3px solid #27ae60; border-radius: 4px;")
        self.ai_reasoning_layout.addWidget(self.vlm_title)
        self.ai_reasoning_layout.addWidget(self.vlm_decision_label)
        self.ai_reasoning_layout.addWidget(self.vlm_explanation_label)
        self.ai_reasoning_layout.addSpacing(15)

        # Feedback Buttons
        self.feedback_buttons_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("✅ Confirm Cheating")
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745; /* Green */
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.confirm_btn.clicked.connect(lambda: self.emit_feedback("confirmed_cheating"))
        self.confirm_btn.setEnabled(False) # Initially disabled

        self.false_pos_btn = QPushButton("❌ False Positive")
        self.false_pos_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545; /* Red */
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.false_pos_btn.clicked.connect(lambda: self.emit_feedback("false_positive"))
        self.false_pos_btn.setEnabled(False) # Initially disabled

        self.feedback_buttons_layout.addWidget(self.confirm_btn)
        self.feedback_buttons_layout.addWidget(self.false_pos_btn)
        self.ai_reasoning_layout.addLayout(self.feedback_buttons_layout)
        self.ai_reasoning_layout.addStretch(1) # Push content to top

    def add_event(self, event_data: dict):
        """Adds a new initial anomaly event to the list in the scroll area."""
        event_frame = QFrame(self.initial_events_group)
        event_frame.setFrameShape(QFrame.Shape.Box)
        event_frame.setLineWidth(1)
        event_frame.setStyleSheet("background-color: #e6f7ff; border-radius: 5px; padding: 5px; margin-bottom: 5px; border: 1px solid #b3d9ff;")
        event_layout = QVBoxLayout(event_frame)
        event_layout.setContentsMargins(5,5,5,5) # Inner padding

        event_id = event_data.get('event_id', f"Event_{event_data.get('timestamp'):.2f}")
        event_layout.addWidget(QLabel(f"<b>Event ID:</b> {event_id}"))
        event_layout.addWidget(QLabel(f"<b>Timestamp:</b> {event_data.get('timestamp'):.2f}s"))
        event_layout.addWidget(QLabel(f"<b>Type:</b> {event_data.get('type').replace('_', ' ').title()}"))
        event_layout.addWidget(QLabel(f"<b>Description:</b> {event_data.get('description')}"))

        process_button = QPushButton("✨ Analyze with AI")
        process_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* Green */
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        process_button.clicked.connect(lambda _, ed=event_data: self.display_ai_reasoning_placeholder(ed)) # Use lambda to pass event_data
        event_layout.addWidget(process_button)

        self.initial_events_group.layout().addWidget(event_frame)
        self.initial_events_group.layout().addStretch(1) # Push content to top

    def display_ai_reasoning_placeholder(self, event_data: dict):
        """
        Sets the UI to 'generating' state and stores active event data.
        This is called first when a user clicks 'Process with AI'.
        """
        self.active_event_data = event_data

        self.current_event_label.setText(f"<b>Analyzing Event:</b> {event_data.get('type').replace('_', ' ').title()} at {event_data.get('timestamp'):.2f}s")
        self.llm_constraint_label.setText("<i>Generating constraints from LLM...</i>")
        self.llm_constraint_label.setStyleSheet("background-color: #ecf0f1; padding: 8px; border-left: 3px solid #3498db; border-radius: 4px;")

        self.vlm_decision_label.setText("<i>Waiting for VLM analysis...</i>")
        self.vlm_decision_label.setStyleSheet("font-weight: bold; font-size: 15px; color: gray;")
        self.vlm_explanation_label.setText("")
        self.vlm_explanation_label.setStyleSheet("background-color: #ecf0f1; padding: 8px; border-left: 3px solid #27ae60; border-radius: 4px;")

        self.confirm_btn.setEnabled(False)
        self.false_pos_btn.setEnabled(False)

    def update_llm_constraint(self, constraint: str):
        """Updates the LLM constraint display."""
        if self.active_event_data: # Only update if an event is active
            self.llm_constraint_label.setText(constraint)

    def update_vlm_result(self, data: dict):
        """Updates the VLM decision and explanation display."""
        # This signal carries all the data necessary for feedback as well
        event_data = data['event_data']
        llm_constraint = data['llm_constraint']
        vlm_result = data['vlm_result']

        if self.active_event_data and self.active_event_data.get('event_id') == event_data.get('event_id'):
            self.vlm_decision_label.setText(vlm_result.get('decision', 'N/A'))
            self.vlm_explanation_label.setText(vlm_result.get('explanation', ''))

            if "Confirmed" in vlm_result.get('decision', ''):
                self.vlm_decision_label.setStyleSheet("font-weight: bold; font-size: 15px; color: #c0392b;") # Red for Confirmed
            elif "Not Cheating" in vlm_result.get('decision', ''):
                self.vlm_decision_label.setStyleSheet("font-weight: bold; font-size: 15px; color: #27ae60;") # Green for Not Cheating
            else:
                self.vlm_decision_label.setStyleSheet("font-weight: bold; font-size: 15px; color: #f39c12;") # Orange for Ambiguous/Error

            self.confirm_btn.setEnabled(True)
            self.false_pos_btn.setEnabled(True)

            # Store VLM decision and explanation within active_event_data for feedback
            self.active_event_data['vlm_decision_text'] = vlm_result.get('decision')
            self.active_event_data['vlm_explanation_text'] = vlm_result.get('explanation')

    def emit_feedback(self, feedback_type: str):
        """Emits a signal when feedback is provided, and disables buttons."""
        if self.active_event_data:
            self.feedback_provided.emit(self.active_event_data, feedback_type)
            self.confirm_btn.setEnabled(False)
            self.false_pos_btn.setEnabled(False)
            self.active_event_data = None # Clear active event after feedback

