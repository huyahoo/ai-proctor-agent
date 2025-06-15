from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QFrame
from PyQt6.QtCore import Qt, pyqtSignal

class AlertPanel(QWidget):
    feedback_provided = pyqtSignal(dict, str) # event_data, feedback_type (confirmed_cheating, false_positive)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignmentFlag.AlignTop)

        self.alert_label = QLabel("Anomaly Detection & AI Reasoning")
        self.alert_label.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 10px;")
        self.layout().addWidget(self.alert_label)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content_widget = QWidget()
        self.scroll_content_layout = QVBoxLayout(self.scroll_content_widget)
        self.scroll_content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_content_widget)
        self.layout().addWidget(self.scroll_area)

        self.active_event_data = None # Store data for the event being displayed/feedbacked

    def add_event(self, event_data: dict):
        """Adds a new initial anomaly event to the panel."""
        event_frame = QFrame(self.scroll_content_widget)
        event_frame.setFrameShape(QFrame.Shape.Box)
        event_frame.setLineWidth(1)
        event_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin-bottom: 5px;")
        event_layout = QVBoxLayout(event_frame)

        event_id = event_data.get('event_id', f"Event_{event_data.get('timestamp'):.2f}")
        event_layout.addWidget(QLabel(f"**Event ID:** {event_id}"))
        event_layout.addWidget(QLabel(f"**Timestamp:** {event_data.get('timestamp'):.2f}s"))
        event_layout.addWidget(QLabel(f"**Type:** {event_data.get('type').replace('_', ' ').title()}"))
        event_layout.addWidget(QLabel(f"**Description:** {event_data.get('description')}"))

        process_button = QPushButton("Process with AI")
        process_button.clicked.connect(lambda: self.display_ai_reasoning(event_data))
        event_layout.addWidget(process_button)

        self.scroll_content_layout.addWidget(event_frame)
        self.scroll_content_layout.addStretch(1) # Push content to top

    def display_ai_reasoning(self, event_data: dict, llm_constraint: str = None, vlm_result: dict = None):
        """
        Displays the AI reasoning process for a selected event.
        Call this with initial event_data, then update with llm_constraint and vlm_result.
        """
        # Clear previous reasoning display
        for i in reversed(range(self.scroll_content_layout.count())):
            widget = self.scroll_content_layout.itemAt(i).widget()
            if widget and widget is not self.scroll_area: # Don't remove the scroll area itself
                widget.setParent(None)
                widget.deleteLater()

        self.active_event_data = event_data # Store current event for feedback

        # Initial Event Info
        event_info_frame = QFrame(self.scroll_content_widget)
        event_info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        event_info_frame.setStyleSheet("background-color: #e0e0e0; border-radius: 5px; padding: 10px; margin-bottom: 10px;")
        event_info_layout = QVBoxLayout(event_info_frame)
        event_info_layout.addWidget(QLabel(f"**Analyzing Event:** {event_data.get('type').replace('_', ' ').title()} at {event_data.get('timestamp'):.2f}s"))
        event_info_layout.addWidget(QLabel(f"Description: {event_data.get('description')}"))
        self.scroll_content_layout.addWidget(event_info_frame)

        # LLM Constraints Display
        llm_frame = QFrame(self.scroll_content_widget)
        llm_frame.setFrameShape(QFrame.Shape.StyledPanel)
        llm_frame.setStyleSheet("background-color: #dbe9f9; border-radius: 5px; padding: 10px; margin-bottom: 10px;")
        llm_layout = QVBoxLayout(llm_frame)
        llm_layout.addWidget(QLabel("<b>LLM Generated Constraint:</b>"))
        self.llm_constraint_label = QLabel(llm_constraint if llm_constraint else "<i>Generating...</i>")
        self.llm_constraint_label.setWordWrap(True)
        llm_layout.addWidget(self.llm_constraint_label)
        self.scroll_content_layout.addWidget(llm_frame)

        # VLM Analysis Display
        vlm_frame = QFrame(self.scroll_content_widget)
        vlm_frame.setFrameShape(QFrame.Shape.StyledPanel)
        vlm_frame.setStyleSheet("background-color: #e0ffe0; border-radius: 5px; padding: 10px; margin-bottom: 10px;")
        vlm_layout = QVBoxLayout(vlm_frame)
        vlm_layout.addWidget(QLabel("<b>VLM Analysis & Decision:</b>"))
        self.vlm_decision_label = QLabel(vlm_result.get('decision', "<i>Analyzing...</i>") if vlm_result else "<i>Analyzing...</i>")
        self.vlm_decision_label.setStyleSheet("font-weight: bold; color: green;")
        self.vlm_explanation_label = QLabel(vlm_result.get('explanation', "") if vlm_result else "")
        self.vlm_explanation_label.setWordWrap(True)
        vlm_layout.addWidget(self.vlm_decision_label)
        vlm_layout.addWidget(self.vlm_explanation_label)
        self.scroll_content_layout.addWidget(vlm_frame)

        if vlm_result:
            if "Confirmed" in vlm_result.get('decision', ''):
                self.vlm_decision_label.setStyleSheet("font-weight: bold; color: red;") # Change color for confirmed cheating
            else:
                self.vlm_decision_label.setStyleSheet("font-weight: bold; color: green;")

            # Feedback Buttons
            feedback_layout = QHBoxLayout()
            confirm_btn = QPushButton("✅ Confirm Cheating")
            confirm_btn.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px; padding: 5px;")
            confirm_btn.clicked.connect(lambda: self.emit_feedback("confirmed_cheating"))
            feedback_layout.addWidget(confirm_btn)

            false_pos_btn = QPushButton("❌ False Positive")
            false_pos_btn.setStyleSheet("background-color: #f44336; color: white; border-radius: 5px; padding: 5px;")
            false_pos_btn.clicked.connect(lambda: self.emit_feedback("false_positive"))
            feedback_layout.addWidget(false_pos_btn)

            self.scroll_content_layout.addLayout(feedback_layout)

    def update_llm_constraint(self, constraint: str):
        if hasattr(self, 'llm_constraint_label'):
            self.llm_constraint_label.setText(constraint)

    def update_vlm_result(self, result: dict):
        if hasattr(self, 'vlm_decision_label'):
            self.vlm_decision_label.setText(result.get('decision', 'N/A'))
            self.vlm_explanation_label.setText(result.get('explanation', ''))
            if "Confirmed" in result.get('decision', ''):
                self.vlm_decision_label.setStyleSheet("font-weight: bold; color: red;")
            else:
                self.vlm_decision_label.setStyleSheet("font-weight: bold; color: green;")

            # Add feedback buttons once VLM result is in
            feedback_layout = QHBoxLayout()
            confirm_btn = QPushButton("✅ Confirm Cheating")
            confirm_btn.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px; padding: 5px;")
            confirm_btn.clicked.connect(lambda: self.emit_feedback("confirmed_cheating"))
            feedback_layout.addWidget(confirm_btn)

            false_pos_btn = QPushButton("❌ False Positive")
            false_pos_btn.setStyleSheet("background-color: #f44336; color: white; border-radius: 5px; padding: 5px;")
            false_pos_btn.clicked.connect(lambda: self.emit_feedback("false_positive"))
            feedback_layout.addWidget(false_pos_btn)

            self.scroll_content_layout.addLayout(feedback_layout)

    def emit_feedback(self, feedback_type: str):
        if self.active_event_data:
            vlm_decision = self.vlm_decision_label.text() if hasattr(self, 'vlm_decision_label') else "N/A"
            vlm_explanation = self.vlm_explanation_label.text() if hasattr(self, 'vlm_explanation_label') else "N/A"
            self.feedback_provided.emit(
                {**self.active_event_data, 'vlm_decision_text': vlm_decision, 'vlm_explanation_text': vlm_explanation},
                feedback_type
            )
            # Disable feedback buttons after submission
            sender_button = self.sender()
            if sender_button:
                parent_layout = sender_button.parentWidget().layout()
                if parent_layout:
                    for i in range(parent_layout.count()):
                        item = parent_layout.itemAt(i)
                        widget = item.widget()
                        if widget and isinstance(widget, QPushButton):
                            widget.setEnabled(False)
        self.active_event_data = None # Clear active event after feedback

