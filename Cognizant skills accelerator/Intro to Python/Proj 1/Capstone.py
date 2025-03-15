import datetime

class StudyPlanner:
    def __init__(self):
        self.tasks = []
        self.scores = {}

    # Task Management
    def add_task(self, task_name, deadline, priority):
        try:
            deadline_date = datetime.datetime.strptime(deadline, "%Y-%m-%d")
            self.tasks.append({"task": task_name, "deadline": deadline_date, "priority": priority})
            self.tasks.sort(key=lambda x: (x["deadline"], -x["priority"]))  # Sort by deadline and priority
            print(f"Task '{task_name}' added successfully!")
        except ValueError:
            print("Invalid date format! Use YYYY-MM-DD.")

    def view_tasks(self):
        if not self.tasks:
            print("No tasks available.")
        else:
            print("\nTask List (Sorted by Deadline and Priority):")
            for task in self.tasks:
                print(f"{task['task']} - Due: {task['deadline'].date()} - Priority: {task['priority']}")

    # Performance Tracking
    def add_score(self, subject, score):
        try:
            score = float(score)
            if subject not in self.scores:
                self.scores[subject] = []
            self.scores[subject].append(score)
            print(f"Score {score} added for {subject}!")
        except ValueError:
            print("Invalid score input. Please enter a numerical value.")

    def view_performance(self):
        if not self.scores:
            print("No scores available.")
            return
        print("\nPerformance Report:")
        for subject, scores in self.scores.items():
            avg_score = sum(scores) / len(scores)
            status = "Needs Improvement" if avg_score < 60 else "Good"
            print(f"{subject}: Avg Score = {avg_score:.2f} ({status})")

    # Main Menu
    def main_menu(self):
        while True:
            print("\nStudy Planner - Main Menu")
            print("1. Add Task")
            print("2. View Tasks")
            print("3. Add Score")
            print("4. View Performance")
            print("5. Exit")
            
            choice = input("Choose an option: ")
            if choice == "1":
                task = input("Enter task name: ")
                deadline = input("Enter deadline (YYYY-MM-DD): ")
                priority = input("Enter priority (1-5): ")
                try:
                    priority = int(priority)
                    if 1 <= priority <= 5:
                        self.add_task(task, deadline, priority)
                    else:
                        print("Priority must be between 1 and 5.")
                except ValueError:
                    print("Invalid priority input.")
            elif choice == "2":
                self.view_tasks()
            elif choice == "3":
                subject = input("Enter subject: ")
                score = input("Enter score: ")
                self.add_score(subject, score)
            elif choice == "4":
                self.view_performance()
            elif choice == "5":
                print("Exiting Study Planner. Goodbye!")
                break
            else:
                print("Invalid option. Please choose again.")


# Run the program
if __name__ == "__main__":
    planner = StudyPlanner()
    planner.main_menu()
